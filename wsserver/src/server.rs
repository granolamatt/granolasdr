use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::dict::Dict;
use crate::protocol::{validate_key, ClientMessage, ServerMessage};
use crate::subs::{key_matches_any, Codec, ConnectionId, Subscriptions, CHANNEL_CAPACITY};

pub struct AppState {
    pub dict: Dict,
    pub subs: Subscriptions,
}

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.max_message_size(16 * 1024 * 1024)
        .on_upgrade(move |socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, state: Arc<AppState>) {
    let conn_id = state.subs.next_id().await;
    tracing::info!(conn_id, "connection established");

    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<Message>(CHANNEL_CAPACITY);

    let hello = serde_json::to_string(&ServerMessage::Hello { version: "1.0" }).expect("serialize");
    if sink.send(Message::Text(hello)).await.is_err() {
        return;
    }

    let mut write_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sink.send(msg).await.is_err() {
                break;
            }
        }
    });

    let mut conn = Connection {
        conn_id,
        tx,
        state: state.clone(),
        subscribed: false,
        push_codec: Codec::Json,
    };

    loop {
        tokio::select! {
            result = stream.next() => {
                match result {
                    Some(Ok(Message::Text(text))) => conn.handle_text(&text).await,
                    Some(Ok(Message::Binary(data))) => conn.handle_binary(&data[..]).await,
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        let _ = conn.tx.send(Message::Pong(data)).await;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::debug!(conn_id, error = %e, "read error");
                        break;
                    }
                }
            }
            _ = &mut write_task => break,
        }
    }

    write_task.abort();
    state.subs.unregister(conn_id).await;
    tracing::info!(conn_id, "connection closed");
}

// TTL sweep — runs forever; caller wraps in a restart loop.
pub async fn run_ttl_sweep(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let expired = state.dict.sweep_expired().await;
        for key in expired {
            let (json_msg, mp_msg) = {
                let msg = ServerMessage::KeyExpired { key: key.as_str() };
                encode_both(&msg)
            };
            state.subs.fan_out(&key, &json_msg, &mp_msg).await;
        }
    }
}

fn encode_for_codec<T: serde::Serialize>(msg: &T, codec: Codec) -> Message {
    match codec {
        Codec::Json => Message::Text(serde_json::to_string(msg).expect("json encode")),
        Codec::Msgpack => {
            Message::Binary(rmp_serde::to_vec_named(msg).expect("msgpack encode").into())
        }
    }
}

fn encode_both<T: serde::Serialize>(msg: &T) -> (Message, Message) {
    let json = Message::Text(serde_json::to_string(msg).expect("json encode"));
    let mp = Message::Binary(rmp_serde::to_vec_named(msg).expect("msgpack encode").into());
    (json, mp)
}

struct Connection {
    conn_id: ConnectionId,
    tx: mpsc::Sender<Message>,
    state: Arc<AppState>,
    subscribed: bool,
    push_codec: Codec,
}

impl Connection {
    async fn handle_text(&mut self, text: &str) {
        let msg: ClientMessage = match serde_json::from_str(text) {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(conn_id = self.conn_id, error = %e, "json parse error");
                self.send_err("invalid JSON", "", None).await;
                return;
            }
        };
        self.dispatch(msg, Codec::Json).await;
    }

    async fn handle_binary(&mut self, data: &[u8]) {
        let msg: ClientMessage = match rmp_serde::from_slice(data) {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(conn_id = self.conn_id, error = %e, "msgpack parse error");
                self.send_err("invalid msgpack", "", None).await;
                return;
            }
        };
        self.dispatch(msg, Codec::Msgpack).await;
    }

    async fn dispatch(&mut self, msg: ClientMessage, frame_codec: Codec) {
        match msg {
            ClientMessage::Subscribe { keys, id } => {
                self.handle_subscribe(keys, id, frame_codec).await
            }
            ClientMessage::Set { key, value, ttl_ms } => {
                self.handle_set(key, value, ttl_ms).await
            }
            ClientMessage::Get { key, id } => self.handle_get(key, id, frame_codec).await,
            ClientMessage::Delete { key, id } => self.handle_delete(key, id).await,
            ClientMessage::List => self.handle_list(frame_codec).await,
        }
    }

    async fn handle_subscribe(
        &mut self,
        keys: Vec<String>,
        id: Option<serde_json::Value>,
        frame_codec: Codec,
    ) {
        if let Err(e) = keys.iter().try_for_each(|k| validate_key(k)) {
            self.send_err("invalid key", &e, id.as_ref()).await;
            return;
        }

        if !self.subscribed {
            self.push_codec = frame_codec;
            self.state
                .subs
                .register(self.conn_id, keys.clone(), self.tx.clone(), frame_codec)
                .await;
            self.subscribed = true;

            let subscribed_msg = ServerMessage::Subscribed { keys: &keys, id: id.as_ref() };
            let _ = self.tx.send(encode_for_codec(&subscribed_msg, self.push_codec)).await;

            self.push_current_values(&keys).await;
        } else {
            let existing = self.state.subs.get_subscribed_keys(self.conn_id).await;
            let new_keys: Vec<String> = keys
                .iter()
                .filter(|k| !existing.contains(*k))
                .cloned()
                .collect();

            self.state.subs.merge_keys(self.conn_id, &keys).await;
            let all_keys = self.state.subs.get_subscribed_keys(self.conn_id).await;

            let subscribed_msg = ServerMessage::Subscribed { keys: &all_keys, id: id.as_ref() };
            let _ = self.tx.send(encode_for_codec(&subscribed_msg, self.push_codec)).await;

            self.push_current_values(&new_keys).await;
        }
    }

    async fn push_current_values(&self, patterns: &[String]) {
        let all_dict_keys = self.state.dict.keys().await;
        for dict_key in &all_dict_keys {
            if key_matches_any(dict_key, patterns) {
                if let Some(value) = self.state.dict.get(dict_key).await {
                    let msg = ServerMessage::Update { key: dict_key, value: &value };
                    let _ = self.tx.send(encode_for_codec(&msg, self.push_codec)).await;
                }
            }
        }
    }

    async fn handle_set(&self, key: String, value: serde_json::Value, ttl_ms: Option<u64>) {
        if let Err(e) = validate_key(&key) {
            self.send_err("invalid key", &e, None).await;
            return;
        }

        if let Some(ttl) = ttl_ms {
            if ttl == 0 {
                self.send_err("ttl_ms must be > 0", &key, None).await;
                return;
            }
            self.state.dict.set_with_ttl(key.clone(), value.clone(), ttl).await;
        } else {
            self.state.dict.set(key.clone(), value.clone()).await;
        }

        let (json_msg, mp_msg) = {
            let msg = ServerMessage::Update { key: &key, value: &value };
            encode_both(&msg)
        };
        self.state.subs.fan_out(&key, &json_msg, &mp_msg).await;
    }

    async fn handle_get(
        &self,
        key: String,
        id: Option<serde_json::Value>,
        frame_codec: Codec,
    ) {
        if let Err(e) = validate_key(&key) {
            self.send_err("invalid key", &e, id.as_ref()).await;
            return;
        }

        match self.state.dict.get(&key).await {
            Some(value) => {
                let msg = ServerMessage::Value { key: &key, value: &value, id: id.as_ref() };
                let _ = self.tx.send(encode_for_codec(&msg, frame_codec)).await;
            }
            None => {
                self.send_err("key not found", &key, id.as_ref()).await;
            }
        }
    }

    async fn handle_delete(&self, key: String, id: Option<serde_json::Value>) {
        if let Err(e) = validate_key(&key) {
            self.send_err("invalid key", &e, None).await;
            return;
        }

        if self.state.dict.delete(&key).await {
            let (json_msg, mp_msg) = {
                let msg = ServerMessage::KeyDeleted { key: &key };
                encode_both(&msg)
            };
            self.state.subs.fan_out(&key, &json_msg, &mp_msg).await;
        } else {
            self.send_err("key not found", &key, id.as_ref()).await;
        }
    }

    async fn handle_list(&self, frame_codec: Codec) {
        let keys = self.state.dict.keys().await;
        let msg = ServerMessage::Keys { keys: &keys };
        let _ = self.tx.send(encode_for_codec(&msg, frame_codec)).await;
    }

    async fn send_err(&self, message: &str, key: &str, id: Option<&serde_json::Value>) {
        let text = serde_json::to_string(&ServerMessage::Error { message, key, id }).unwrap();
        let _ = self.tx.send(Message::Text(text)).await;
    }
}
