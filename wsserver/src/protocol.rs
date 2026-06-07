use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "subscribe")]
    Subscribe {
        keys: Vec<String>,
        #[serde(default)]
        id: Option<serde_json::Value>,
    },
    #[serde(rename = "set")]
    Set {
        key: String,
        value: serde_json::Value,
        #[serde(default)]
        ttl_ms: Option<u64>,
    },
    #[serde(rename = "list")]
    List,
    #[serde(rename = "get")]
    Get {
        key: String,
        #[serde(default)]
        id: Option<serde_json::Value>,
    },
    #[serde(rename = "delete")]
    Delete {
        key: String,
        #[serde(default)]
        id: Option<serde_json::Value>,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMessage<'a> {
    #[serde(rename = "hello")]
    Hello { version: &'a str },
    #[serde(rename = "subscribed")]
    Subscribed {
        keys: &'a [String],
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<&'a serde_json::Value>,
    },
    #[serde(rename = "update")]
    Update {
        key: &'a str,
        value: &'a serde_json::Value,
    },
    #[serde(rename = "key_deleted")]
    KeyDeleted { key: &'a str },
    #[serde(rename = "key_expired")]
    KeyExpired { key: &'a str },
    #[serde(rename = "keys")]
    Keys { keys: &'a [String] },
    #[serde(rename = "value")]
    Value {
        key: &'a str,
        value: &'a serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<&'a serde_json::Value>,
    },
    #[serde(rename = "error")]
    Error {
        message: &'a str,
        key: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<&'a serde_json::Value>,
    },
}

pub fn validate_key(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("empty key".into());
    }
    if key.trim().is_empty() {
        return Err("whitespace-only key".into());
    }
    if key.contains('\0') {
        return Err("null byte in key".into());
    }
    if key.len() > 1024 {
        return Err("key too long (max 1024 bytes)".into());
    }
    Ok(())
}
