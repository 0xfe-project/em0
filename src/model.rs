//! Value types for the memory API.

/// Scoping: every memory belongs to exactly one (user, session, agent) triple.
/// Empty string is the default slot; all three must match for visibility.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Scope {
    pub user_id: String,
    pub session_id: String,
    pub agent_id: String,
}

impl Scope {
    pub fn user(user_id: &str) -> Self {
        Scope {
            user_id: user_id.to_string(),
            session_id: String::new(),
            agent_id: String::new(),
        }
    }

    pub fn new(user_id: &str, session_id: &str, agent_id: &str) -> Self {
        Scope {
            user_id: user_id.to_string(),
            session_id: session_id.to_string(),
            agent_id: agent_id.to_string(),
        }
    }
}

/// Tunables for the deterministic dedup in `add` (cosine thresholds).
#[derive(Debug, Clone, Copy)]
pub struct Em0Options {
    /// above this cosine the memory is considered a duplicate -> NOOP
    pub dedup_noop: f32,
    /// above this (and <= dedup_noop) the memory is a rewrite -> UPDATE
    pub dedup_update: f32,
}

impl Default for Em0Options {
    fn default() -> Self {
        Em0Options {
            dedup_noop: 0.95,
            dedup_update: 0.80,
        }
    }
}

/// Outcome of `add`.
#[derive(Debug, Clone, PartialEq)]
pub enum AddEvent {
    Add { id: String },
    Update { id: String },
    Noop { id: String },
}

/// One search result. `score` is cosine for `search_vec`, the RRF score for
/// hybrid `search` (with the vector-path cosine in `cosine` when present).
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub cosine: Option<f32>,
    pub metadata: serde_json::Value,
}

/// A stored memory, as returned by `get_all`.
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub scope: Scope,
    pub metadata: serde_json::Value,
    pub created_at: i64,
    pub updated_at: i64,
}

/// One entry of the append-only audit trail.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEvent {
    pub seq: i64,
    pub memory_id: String,
    pub event: String, // ADD | UPDATE | NOOP | DELETE
    pub old_content: Option<String>,
    pub new_content: Option<String>,
    pub ts: i64,
}
