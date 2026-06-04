use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::dict::Dict;
use crate::protocol::{validate_key, ClientMessage, ServerMessage};
use crate::subs::{ConnectionId, Subscriptions, CHANNEL_CAPACITY};

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
    };

    loop {
        tokio::select! {
            result = stream.next() => {
                match result {
                    Some(Ok(Message::Text(text))) => conn.handle_text(&text).await,
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

struct Connection {
    conn_id: ConnectionId,
    tx: mpsc::Sender<Message>,
    state: Arc<AppState>,
    subscribed: bool,
}

impl Connection {
    async fn handle_text(&mut self, text: &str) {
        let msg: ClientMessage = match serde_json::from_str(text) {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(conn_id = self.conn_id, error = %e, "parse error");
                self.send_err("invalid JSON", "", None).await;
                return;
            }
        };

        match msg {
            ClientMessage::Subscribe { keys, id } => self.handle_subscribe(keys, id).await,
            ClientMessage::Set { key, value } => self.handle_set(key, value).await,
            ClientMessage::Get { key, id } => self.handle_get(key, id).await,
            ClientMessage::Delete { key, id } => self.handle_delete(key, id).await,
        }
    }

    async fn handle_subscribe(&mut self, keys: Vec<String>, id: Option<serde_json::Value>) {
        if let Err(e) = keys.iter().try_for_each(|k| validate_key(k)) {
            self.send_err("invalid key", &e, id.as_ref()).await;
            return;
        }

        if !self.subscribed {
            self.state
                .subs
                .register(self.conn_id, keys.clone(), self.tx.clone())
                .await;
            self.subscribed = true;

            let text = serde_json::to_string(&ServerMessage::Subscribed {
                keys: &keys,
                id: id.as_ref(),
            })
            .unwrap();
            let _ = self.tx.send(Message::Text(text)).await;

            for key in &keys {
                if let Some(value) = self.state.dict.get(key).await {
                    let text = serde_json::to_string(&ServerMessage::Update {
                        key,
                        value: &value,
                    })
                    .unwrap();
                    let _ = self.tx.send(Message::Text(text)).await;
                }
            }
        } else {
            let existing = self.state.subs.get_subscribed_keys(self.conn_id).await;
            let new_keys: Vec<String> = keys
                .iter()
                .filter(|k| !existing.contains(*k))
                .cloned()
                .collect();

            self.state.subs.merge_keys(self.conn_id, &keys).await;
            let all_keys = self.state.subs.get_subscribed_keys(self.conn_id).await;
            let text = serde_json::to_string(&ServerMessage::Subscribed {
                keys: &all_keys,
                id: id.as_ref(),
            })
            .unwrap();
            let _ = self.tx.send(Message::Text(text)).await;

            for key in &new_keys {
                if let Some(value) = self.state.dict.get(key).await {
                    let text = serde_json::to_string(&ServerMessage::Update {
                        key,
                        value: &value,
                    })
                    .unwrap();
                    let _ = self.tx.send(Message::Text(text)).await;
                }
            }
        }
    }

    async fn handle_set(&self, key: String, value: serde_json::Value) {
        if let Err(e) = validate_key(&key) {
            self.send_err("invalid key", &e, None).await;
            return;
        }

        self.state.dict.set(key.clone(), value.clone()).await;

        let update = serde_json::to_string(&ServerMessage::Update {
            key: &key,
            value: &value,
        })
        .unwrap();
        self.state
            .subs
            .fan_out(&key, &Message::Text(update))
            .await;
    }

    async fn handle_get(&self, key: String, id: Option<serde_json::Value>) {
        if let Err(e) = validate_key(&key) {
            self.send_err("invalid key", &e, id.as_ref()).await;
            return;
        }

        match self.state.dict.get(&key).await {
            Some(value) => {
                let text = serde_json::to_string(&ServerMessage::Value {
                    key: &key,
                    value: &value,
                    id: id.as_ref(),
                })
                .unwrap();
                let _ = self.tx.send(Message::Text(text)).await;
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
            let del = serde_json::to_string(&ServerMessage::KeyDeleted { key: &key }).unwrap();
            self.state
                .subs
                .fan_out(&key, &Message::Text(del))
                .await;
        } else {
            self.send_err("key not found", &key, id.as_ref()).await;
        }
    }

    async fn send_err(&self, message: &str, key: &str, id: Option<&serde_json::Value>) {
        let text = serde_json::to_string(&ServerMessage::Error { message, key, id }).unwrap();
        let _ = self.tx.send(Message::Text(text)).await;
    }
}
