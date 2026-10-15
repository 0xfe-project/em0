//! em0 storage: one sqlite file IS the whole memory store.
//!
//! `memories` carries content + scope + metadata; `memories_vec` (vec0)
//! carries the 640-dim L2-normalized embeddings joined by rowid; `history`
//! is the append-only audit trail. rusqlite(bundled) + sqlite-vec loaded
//! per connection; the connection lives behind a Mutex (rusqlite is !Sync).

use crate::harrier::Harrier;
use rusqlite::Connection;
use std::fmt;
use std::path::Path;
use std::sync::{Mutex, Once};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const MIGRATION: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    user_id    TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    agent_id   TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(user_id, session_id, agent_id);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    embedding FLOAT[640],
    user_id TEXT,
    session_id TEXT,
    agent_id TEXT
);
CREATE TABLE IF NOT EXISTS history (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   TEXT NOT NULL,
    event       TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    ts          INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
";

#[derive(Debug)]
pub enum Em0Error {
    Sqlite(rusqlite::Error),
    Harrier(crate::harrier::Error),
}

impl fmt::Display for Em0Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Em0Error::Sqlite(e) => write!(f, "sqlite: {e}"),
            Em0Error::Harrier(e) => write!(f, "harrier: {e}"),
        }
    }
}
impl std::error::Error for Em0Error {}
impl From<rusqlite::Error> for Em0Error {
    fn from(e: rusqlite::Error) -> Self {
        Em0Error::Sqlite(e)
    }
}

pub(crate) fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// f32 slice -> little-endian blob (vec0 storage format).
pub(crate) fn f32_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Register sqlite-vec for every future connection (must run BEFORE the first
/// Connection::open; sqlite-vec is compiled into the binary, no dylib).
fn ensure_vec_ext() {
    static VEC_EXT: Once = Once::new();
    VEC_EXT.call_once(|| unsafe {
        // sqlite3_vec_init is declared 0-arg in the crate; the entry-point
        // signature sqlite3_auto_extension expects is the 3-arg extension init.
        let init: unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::os::raw::c_int =
            std::mem::transmute::<*const (), _>(sqlite_vec::sqlite3_vec_init as *const ());
        rusqlite::ffi::sqlite3_auto_extension(Some(init));
    });
}

pub struct Em0 {
    pub(crate) db: Mutex<Connection>,
    pub(crate) h: Harrier,
    pub(crate) opts: crate::model::Em0Options,
}

impl Em0 {
    /// Open (and migrate) a memory store at `path`. One file is everything.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Em0Error> {
        Self::open_with(path, crate::model::Em0Options::default())
    }

    pub fn open_with(
        path: impl AsRef<Path>,
        opts: crate::model::Em0Options,
    ) -> Result<Self, Em0Error> {
        ensure_vec_ext();
        let conn = Connection::open(path)?;
        Self::init(conn, opts)
    }

    /// In-memory store: tests and ephemeral sessions.
    pub fn open_in_memory() -> Result<Self, Em0Error> {
        ensure_vec_ext();
        Self::init(
            Connection::open_in_memory()?,
            crate::model::Em0Options::default(),
        )
    }

    fn init(conn: Connection, opts: crate::model::Em0Options) -> Result<Self, Em0Error> {
        // extension must be live before any vec0 DDL
        let _: String = conn.query_row("SELECT vec_version()", [], |r| r.get(0))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;",
        )?;
        conn.execute_batch(MIGRATION)?;
        Ok(Em0 {
            db: Mutex::new(conn),
            h: Harrier::new().map_err(Em0Error::Harrier)?,
            opts,
        })
    }
}
