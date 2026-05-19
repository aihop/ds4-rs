use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::SessionSnapshot;

#[derive(Clone, Debug)]
pub struct ContinuationEntry {
    pub response_id: String,
    pub visible_text: String,
    pub conversation: Option<String>,
    pub last_tool_call_id: Option<String>,
    pub snapshot: SessionSnapshot,
    pub created_at_ms: u128,
    pub last_used_ms: u128,
}

impl ContinuationEntry {
    pub fn new(
        response_id: impl Into<String>,
        visible_text: impl Into<String>,
        conversation: Option<String>,
        last_tool_call_id: Option<String>,
        snapshot: SessionSnapshot,
    ) -> Self {
        let now = now_ms();
        Self {
            response_id: response_id.into(),
            visible_text: visible_text.into(),
            conversation,
            last_tool_call_id,
            snapshot,
            created_at_ms: now,
            last_used_ms: now,
        }
    }
}

#[derive(Default)]
pub struct ContinuationStore {
    by_id: HashMap<String, ContinuationEntry>,
    latest_by_conversation: HashMap<String, String>,
    latest_by_tool_call: HashMap<String, String>,
}

impl ContinuationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn remember(&mut self, entry: ContinuationEntry) {
        if let Some(conversation) = &entry.conversation {
            self.latest_by_conversation
                .insert(conversation.clone(), entry.response_id.clone());
        }
        if let Some(tool_call_id) = &entry.last_tool_call_id {
            self.latest_by_tool_call
                .insert(tool_call_id.clone(), entry.response_id.clone());
        }
        self.by_id.insert(entry.response_id.clone(), entry);
    }

    pub fn restore(
        &mut self,
        previous_response_id: Option<&str>,
        conversation: Option<&str>,
        last_tool_call_id: Option<&str>,
    ) -> Option<ContinuationEntry> {
        let entry = if let Some(id) = previous_response_id {
            self.by_id.get_mut(id)
        } else if let Some(conversation) = conversation {
            let latest = self.latest_by_conversation.get(conversation)?.clone();
            self.by_id.get_mut(&latest)
        } else if let Some(tool_call_id) = last_tool_call_id {
            let latest = self.latest_by_tool_call.get(tool_call_id)?.clone();
            self.by_id.get_mut(&latest)
        } else {
            None
        }?;
        entry.last_used_ms = now_ms();
        Some(entry.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_from_latest_tool_call_alias() {
        let mut store = ContinuationStore::new();
        store.remember(ContinuationEntry::new(
            "resp_1",
            "visible",
            None,
            Some("call_1".to_string()),
            SessionSnapshot { bytes: vec![1, 2, 3] },
        ));

        let restored = store.restore(None, None, Some("call_1")).unwrap();
        assert_eq!(restored.response_id, "resp_1");
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}
