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
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_subscribe() {
        let json = r#"{"type":"subscribe","keys":["a","b"]}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Subscribe { keys, id } => {
                assert_eq!(keys, vec!["a", "b"]);
                assert!(id.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_subscribe_with_id() {
        let json = r#"{"type":"subscribe","keys":["a"],"id":"req-1"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Subscribe { keys, id } => {
                assert_eq!(keys, vec!["a"]);
                assert_eq!(id.unwrap(), serde_json::Value::String("req-1".into()));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_set() {
        let json = r#"{"type":"set","key":"x","value":{"nested":42}}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Set { key, value } => {
                assert_eq!(key, "x");
                assert_eq!(value["nested"], 42);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_set_with_null_value() {
        let json = r#"{"type":"set","key":"x","value":null}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Set { key, value } => {
                assert_eq!(key, "x");
                assert!(value.is_null());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_get() {
        let json = r#"{"type":"get","key":"y"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Get { key, .. } => assert_eq!(key, "y"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_delete() {
        let json = r#"{"type":"delete","key":"z"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            ClientMessage::Delete { key, .. } => assert_eq!(key, "z"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_unknown_type() {
        let json = r#"{"type":"unknown","foo":1}"#;
        let err = serde_json::from_str::<ClientMessage>(json).unwrap_err();
        assert!(err.to_string().contains("unknown"));
    }

    #[test]
    fn parse_missing_required_field() {
        let json = r#"{"type":"set"}"#;
        let err = serde_json::from_str::<ClientMessage>(json).unwrap_err();
        assert!(err.to_string().contains("key"));
    }

    #[test]
    fn parse_invalid_json() {
        let json = "not json";
        let err = serde_json::from_str::<ClientMessage>(json).unwrap_err();
        assert!(err.to_string().contains("expected"));
    }

    #[test]
    fn serialize_hello() {
        let msg = ServerMessage::Hello { version: "1.0" };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"hello","version":"1.0"}"#);
    }

    #[test]
    fn serialize_update() {
        let val = serde_json::json!({"x": 1});
        let msg = ServerMessage::Update {
            key: "plot",
            value: &val,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"update","key":"plot","value":{"x":1}}"#);
    }

    #[test]
    fn serialize_key_deleted() {
        let msg = ServerMessage::KeyDeleted { key: "plot" };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"type":"key_deleted","key":"plot"}"#);
    }

    #[test]
    fn serialize_value_with_id() {
        let val = serde_json::json!(42);
        let id = serde_json::Value::Number(serde_json::Number::from(1));
        let msg = ServerMessage::Value {
            key: "x",
            value: &val,
            id: Some(&id),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"value""#));
        assert!(json.contains(r#""key":"x""#));
        assert!(json.contains(r#""value":42"#));
        assert!(json.contains(r#""id":1"#));
    }

    #[test]
    fn serialize_error() {
        let id = serde_json::Value::String("req-1".into());
        let msg = ServerMessage::Error {
            message: "key not found",
            key: "x",
            id: Some(&id),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""message":"key not found""#));
        assert!(json.contains(r#""key":"x""#));
        assert!(json.contains(r#""id":"req-1""#));
    }

    #[test]
    fn subscribe_round_trip() {
        let original = r#"{"type":"subscribe","keys":["a","b/c"],"id":1}"#;
        let msg: ClientMessage = serde_json::from_str(original).unwrap();
        let round = serde_json::to_string(&msg).unwrap();
        let parsed: ClientMessage = serde_json::from_str(&round).unwrap();
        match parsed {
            ClientMessage::Subscribe { keys, .. } => {
                assert_eq!(keys, vec!["a", "b/c"]);
            }
            _ => panic!("round trip failed"),
        }
    }

    #[test]
    fn validate_key_empty() {
        assert!(validate_key("").is_err());
    }

    #[test]
    fn validate_key_whitespace() {
        assert!(validate_key("   ").is_err());
    }

    #[test]
    fn validate_key_null_byte() {
        assert!(validate_key("ab\0c").is_err());
    }

    #[test]
    fn validate_key_too_long() {
        let long = "a".repeat(1025);
        assert!(validate_key(&long).is_err());
    }

    #[test]
    fn validate_key_valid() {
        assert!(validate_key("system/status").is_ok());
        assert!(validate_key("a").is_ok());
        assert!(validate_key("events/*").is_ok());
    }
}
