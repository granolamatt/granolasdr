use axum::extract::ws::Message;
use std::collections::HashMap;
use tokio::sync::mpsc;

pub type ConnectionId = usize;

pub const CHANNEL_CAPACITY: usize = 256;

struct Subscription {
    keys: Vec<String>,
    tx: mpsc::Sender<Message>,
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
    ) {
        let mut subs = self.subs.write().await;
        subs.insert(conn_id, Subscription { keys, tx });
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

    pub async fn fan_out(&self, key: &str, msg: &Message) {
        let subs = self.subs.read().await;
        let mut stale_ids = Vec::new();

        for (conn_id, sub) in subs.iter() {
            if key_matches_any(key, &sub.keys) {
                match sub.tx.try_send(msg.clone()) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        tracing::warn!(conn_id, key, "subscriber channel full, dropping");
                        stale_ids.push(*conn_id);
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        stale_ids.push(*conn_id);
                    }
                }
            }
        }

        drop(subs);

        if !stale_ids.is_empty() {
            let mut subs = self.subs.write().await;
            for id in stale_ids {
                subs.remove(&id);
            }
        }
    }
}

fn key_matches_any(key: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pat| key_matches(key, pat))
}

fn key_matches(key: &str, pattern: &str) -> bool {
    if pattern == key {
        return true;
    }
    if let Some(stem) = pattern.strip_suffix('*') {
        return key.starts_with(stem) && !key[stem.len()..].contains('/');
    }
    false
}

impl Default for Subscriptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        assert!(key_matches("plot_data", "plot_data"));
        assert!(!key_matches("plot_data", "other"));
    }

    #[test]
    fn star_matches_single_segment() {
        assert!(key_matches("foo", "*"));
        assert!(key_matches("plot_data", "*"));
        assert!(!key_matches("foo/bar", "*"));
        assert!(!key_matches("foo/bar/baz", "*"));
    }

    #[test]
    fn wildcard_suffix_star() {
        assert!(key_matches("foo/bar", "foo/*"));
        assert!(!key_matches("foo", "foo/*"));
        assert!(!key_matches("foo/bar/baz", "foo/*"));
        assert!(key_matches("events/a", "events/*"));
        assert!(!key_matches("events/a/b", "events/*"));
    }

    #[test]
    fn wildcard_prefix_star() {
        assert!(key_matches("plot_data", "plot_*"));
        assert!(!key_matches("other_data", "plot_*"));
    }

    #[test]
    fn key_matches_any_checks_all_patterns() {
        let patterns: Vec<String> = vec!["events/*".into(), "system/status".into()];
        assert!(key_matches_any("events/click", &patterns));
        assert!(key_matches_any("system/status", &patterns));
        assert!(!key_matches_any("other/key", &patterns));
    }

    #[tokio::test]
    async fn register_and_get_keys() {
        let subs = Subscriptions::new();
        let id = subs.next_id().await;
        let (tx, _rx) = mpsc::channel(CHANNEL_CAPACITY);
        subs.register(id, vec!["a".into(), "b".into()], tx).await;
        let keys = subs.get_subscribed_keys(id).await;
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn unregister_removes_subscriber() {
        let subs = Subscriptions::new();
        let id = subs.next_id().await;
        let (tx, _rx) = mpsc::channel(CHANNEL_CAPACITY);
        subs.register(id, vec!["a".into()], tx).await;
        subs.unregister(id).await;
        let keys = subs.get_subscribed_keys(id).await;
        assert!(keys.is_empty());
    }

    #[tokio::test]
    async fn merge_keys_adds_new() {
        let subs = Subscriptions::new();
        let id = subs.next_id().await;
        let (tx, _rx) = mpsc::channel(CHANNEL_CAPACITY);
        subs.register(id, vec!["a".into()], tx).await;
        subs.merge_keys(id, &["b".into(), "c".into()]).await;
        let keys = subs.get_subscribed_keys(id).await;
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn fan_out_sends_to_matching_subscribers() {
        let subs = Subscriptions::new();
        let id = subs.next_id().await;
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        subs.register(id, vec!["events/*".into()], tx).await;

        let msg = Message::Text("test".into());
        subs.fan_out("events/click", &msg).await;

        let received = rx.recv().await;
        assert!(received.is_some());
    }

    #[tokio::test]
    async fn fan_out_skips_non_matching_subscribers() {
        let subs = Subscriptions::new();
        let id = subs.next_id().await;
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        subs.register(id, vec!["events/*".into()], tx).await;

        let msg = Message::Text("{}".into());
        subs.fan_out("system/status", &msg).await;

        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn fan_out_removes_closed_subscriber() {
        let subs = Subscriptions::new();
        let id = subs.next_id().await;
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        subs.register(id, vec!["a".into()], tx).await;
        drop(rx);

        let msg = Message::Text("{}".into());
        subs.fan_out("a", &msg).await;

        let keys = subs.get_subscribed_keys(id).await;
        assert!(keys.is_empty());
    }
}
