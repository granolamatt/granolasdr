use axum::extract::ws::Message;
use std::collections::HashMap;
use tokio::sync::mpsc;

pub type ConnectionId = usize;

// Large enough to absorb a full 15s flush burst (100+ decodes) plus continuous
// waterfall updates without dropping the subscriber.
pub const CHANNEL_CAPACITY: usize = 2048;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    Json,
    Msgpack,
}

struct Subscription {
    keys: Vec<String>,
    tx: mpsc::Sender<Message>,
    push_codec: Codec,
}

pub struct Subscriptions {
    subs: tokio::sync::RwLock<HashMap<ConnectionId, Subscription>>,
    next_id: tokio::sync::Mutex<ConnectionId>,
}

impl Subscriptions {
    pub fn new() -> Self {
        Self {
            subs: tokio::sync::RwLock::new(HashMap::new()),
            next_id: tokio::sync::Mutex::new(0),
        }
    }

    pub async fn next_id(&self) -> ConnectionId {
        let mut id = self.next_id.lock().await;
        let current = *id;
        *id = id.wrapping_add(1);
        current
    }

    pub async fn register(
        &self,
        conn_id: ConnectionId,
        keys: Vec<String>,
        tx: mpsc::Sender<Message>,
        push_codec: Codec,
    ) {
        let mut subs = self.subs.write().await;
        subs.insert(conn_id, Subscription { keys, tx, push_codec });
    }

    pub async fn unregister(&self, conn_id: ConnectionId) {
        let mut subs = self.subs.write().await;
        subs.remove(&conn_id);
    }

    pub async fn merge_keys(&self, conn_id: ConnectionId, new_keys: &[String]) {
        let mut subs = self.subs.write().await;
        if let Some(sub) = subs.get_mut(&conn_id) {
            for k in new_keys {
                if !sub.keys.contains(k) {
                    sub.keys.push(k.clone());
                }
            }
        }
    }

    pub async fn get_subscribed_keys(&self, conn_id: ConnectionId) -> Vec<String> {
        let subs = self.subs.read().await;
        subs.get(&conn_id)
            .map(|sub| sub.keys.clone())
            .unwrap_or_default()
    }

    // Fan out to all subscribers whose patterns match `key`.
    // json_msg sent to JSON subscribers, mp_msg to MessagePack subscribers.
    // On channel-full: drops this message but keeps the subscriber (does not disconnect).
    pub async fn fan_out(&self, key: &str, json_msg: &Message, mp_msg: &Message) {
        let subs = self.subs.read().await;
        let mut closed_ids = Vec::new();

        for (conn_id, sub) in subs.iter() {
            if key_matches_any(key, &sub.keys) {
                let msg = match sub.push_codec {
                    Codec::Json => json_msg,
                    Codec::Msgpack => mp_msg,
                };
                match sub.tx.try_send(msg.clone()) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        // Drop this message but keep the subscriber alive.
                        tracing::debug!(conn_id, key, "subscriber channel full, dropping message");
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        closed_ids.push(*conn_id);
                    }
                }
            }
        }

        drop(subs);

        if !closed_ids.is_empty() {
            let mut subs = self.subs.write().await;
            for id in closed_ids {
                subs.remove(&id);
            }
        }
    }
}

pub fn key_matches_any(key: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pat| key_matches(key, pat))
}

pub fn key_matches(key: &str, pattern: &str) -> bool {
    if pattern == key {
        return true;
    }

    // Multi-segment wildcard: foo/** matches foo/bar and foo/bar/baz but NOT bare foo.
    if let Some(stem) = pattern.strip_suffix("/**") {
        return key.starts_with(&format!("{}/", stem));
    }
    if pattern == "**" {
        return true;
    }

    // Single-segment suffix wildcard: foo/* matches foo/bar but not foo/bar/baz.
    if let Some(stem) = pattern.strip_suffix("/*") {
        let rest = key.strip_prefix(&format!("{}/", stem)).unwrap_or("");
        return !rest.is_empty() && !rest.contains('/');
    }

    // Prefix wildcard within a single segment: plot_* matches plot_data but not plot/data.
    if let Some(stem) = pattern.strip_suffix('*') {
        if !stem.contains('/') {
            return key.starts_with(stem) && !key[stem.len()..].contains('/');
        }
    }

    false
}

impl Default for Subscriptions {
    fn default() -> Self {
        Self::new()
    }
}
