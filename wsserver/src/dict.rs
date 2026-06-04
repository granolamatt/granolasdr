use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;

// Keys persisted to disk so they survive pipeline restart via execv.
const PERSISTENT_KEYS: &[&str] = &["uhd:config", "uhd:channels"];

pub struct Dict {
    data: RwLock<HashMap<String, Value>>,
    state_file: Option<PathBuf>,
}

impl Dict {
    pub fn new() -> Self {
        Self { data: RwLock::new(HashMap::new()), state_file: None }
    }

    pub fn new_with_state(path: Option<PathBuf>) -> Self {
        let mut initial = HashMap::new();
        if let Some(ref p) = path {
            if let Ok(contents) = std::fs::read_to_string(p) {
                if let Ok(map) = serde_json::from_str::<HashMap<String, Value>>(&contents) {
                    initial = map;
                    tracing::info!("loaded dict state from {}", p.display());
                }
            }
        }
        Self { data: RwLock::new(initial), state_file: path }
    }

    pub async fn set(&self, key: String, value: Value) {
        let is_persistent = PERSISTENT_KEYS.contains(&key.as_str());
        {
            let mut map = self.data.write().await;
            map.insert(key, value);
        }
        if is_persistent {
            self.persist().await;
        }
    }

    pub async fn get(&self, key: &str) -> Option<Value> {
        let map = self.data.read().await;
        map.get(key).cloned()
    }

    pub async fn delete(&self, key: &str) -> bool {
        let is_persistent = PERSISTENT_KEYS.contains(&key);
        let removed = {
            let mut map = self.data.write().await;
            map.remove(key).is_some()
        };
        if removed && is_persistent {
            self.persist().await;
        }
        removed
    }

    async fn persist(&self) {
        let Some(ref path) = self.state_file else { return };
        let snapshot: HashMap<String, Value> = {
            let map = self.data.read().await;
            PERSISTENT_KEYS.iter()
                .filter_map(|k| map.get(*k).map(|v| ((*k).to_string(), v.clone())))
                .collect()
        };
        if let Ok(json) = serde_json::to_string_pretty(&snapshot) {
            if let Err(e) = tokio::fs::write(path, json).await {
                tracing::warn!("failed to persist dict state to {}: {e}", path.display());
            }
        }
    }
}

impl Default for Dict {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn set_and_get() {
        let dict = Dict::new();
        dict.set("x".into(), json!(42)).await;
        assert_eq!(dict.get("x").await, Some(json!(42)));
    }

    #[tokio::test]
    async fn set_overwrites() {
        let dict = Dict::new();
        dict.set("x".into(), json!(1)).await;
        dict.set("x".into(), json!(2)).await;
        assert_eq!(dict.get("x").await, Some(json!(2)));
    }

    #[tokio::test]
    async fn get_nonexistent() {
        let dict = Dict::new();
        assert_eq!(dict.get("nope").await, None);
    }

    #[tokio::test]
    async fn delete_existing() {
        let dict = Dict::new();
        dict.set("x".into(), json!(42)).await;
        assert!(dict.delete("x").await);
        assert_eq!(dict.get("x").await, None);
    }

    #[tokio::test]
    async fn delete_nonexistent() {
        let dict = Dict::new();
        assert!(!dict.delete("nope").await);
    }

    #[tokio::test]
    async fn set_null_preserves_null() {
        let dict = Dict::new();
        dict.set("x".into(), Value::Null).await;
        assert_eq!(dict.get("x").await, Some(Value::Null));
    }

    #[tokio::test]
    async fn concurrent_writes_no_panic() {
        use std::sync::Arc;
        let dict = Arc::new(Dict::new());
        let mut handles = vec![];

        for i in 0..100 {
            let d = dict.clone();
            handles.push(tokio::spawn(async move {
                d.set("counter".into(), json!(i)).await;
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let val = dict.get("counter").await;
        assert!(val.is_some());
    }
}
