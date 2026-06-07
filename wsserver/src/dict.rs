use serde_json::Value;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

struct Entry {
    value: Value,
    expires_at: Option<Instant>,
}

pub struct Dict {
    data: RwLock<HashMap<String, Entry>>,
}

impl Dict {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    pub async fn set(&self, key: String, value: Value) {
        let mut map = self.data.write().await;
        map.insert(key, Entry { value, expires_at: None });
    }

    pub async fn set_with_ttl(&self, key: String, value: Value, ttl_ms: u64) {
        let expires_at = Instant::now() + Duration::from_millis(ttl_ms);
        let mut map = self.data.write().await;
        map.insert(key, Entry { value, expires_at: Some(expires_at) });
    }

    pub async fn get(&self, key: &str) -> Option<Value> {
        let map = self.data.read().await;
        map.get(key).and_then(|e| {
            if e.expires_at.map_or(true, |t| t > Instant::now()) {
                Some(e.value.clone())
            } else {
                None
            }
        })
    }

    pub async fn delete(&self, key: &str) -> bool {
        let mut map = self.data.write().await;
        map.remove(key).is_some()
    }

    // Returns all non-expired key names.
    pub async fn keys(&self) -> Vec<String> {
        let now = Instant::now();
        let map = self.data.read().await;
        map.iter()
            .filter(|(_, e)| e.expires_at.map_or(true, |t| t > now))
            .map(|(k, _)| k.clone())
            .collect()
    }

    // Two-phase expiry sweep: collect candidates under read lock, re-check under write lock.
    pub async fn sweep_expired(&self) -> Vec<String> {
        let now = Instant::now();
        let candidates: Vec<String> = {
            let map = self.data.read().await;
            map.iter()
                .filter(|(_, e)| e.expires_at.map_or(false, |t| t <= now))
                .map(|(k, _)| k.clone())
                .collect()
        };
        let mut removed = Vec::new();
        for key in candidates {
            let mut map = self.data.write().await;
            if let Some(e) = map.get(&key) {
                if e.expires_at.map_or(false, |t| t <= Instant::now()) {
                    map.remove(&key);
                    removed.push(key);
                }
            }
        }
        removed
    }
}

impl Default for Dict {
    fn default() -> Self {
        Self::new()
    }
}
