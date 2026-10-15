//! The two core operations: `add` (deterministic dedup) and `search`.

use crate::db::{Em0, Em0Error, f32_blob, now};
use crate::model::{AddEvent, HistoryEvent, MemoryRecord, Scope, SearchHit};
use rusqlite::params;
use uuid::Uuid;

/// cosine from vec0 L2 distance over L2-normalized vectors: d² = 2 - 2cos
#[inline]
fn dist_to_cos(d: f64) -> f32 {
    (1.0 - d * d / 2.0) as f32
}

impl Em0 {
    /// Store a memory with deterministic dedup (no LLM): cosine against the
    /// nearest same-scope memory decides NOOP (> `dedup_noop`), UPDATE
    /// (> `dedup_update`), or ADD. Embeddings use the document side (no
    /// instruction prefix).
    pub fn add(
        &self,
        content: &str,
        scope: &Scope,
        metadata: Option<&serde_json::Value>,
    ) -> Result<AddEvent, Em0Error> {
        let emb = self.h.embed(&[content]).map_err(Em0Error::Harrier)?[0];
        self.add_embedded(&emb, content, scope, metadata)
    }

    /// Batch add: embeddings computed in parallel (embed_par), writes are
    /// serialized through the connection lock. Each item runs the same
    /// deterministic dedup as `add`, in order.
    pub fn add_batch(
        &self,
        contents: &[&str],
        scope: &Scope,
        metadata: Option<&serde_json::Value>,
    ) -> Result<Vec<AddEvent>, Em0Error> {
        let embs = self.h.embed_par(contents).map_err(Em0Error::Harrier)?;
        let mut out = Vec::with_capacity(contents.len());
        for (emb, content) in embs.iter().zip(contents) {
            out.push(self.add_embedded(emb, content, scope, metadata)?);
        }
        Ok(out)
    }

    fn add_embedded(
        &self,
        emb: &[f32],
        content: &str,
        scope: &Scope,
        metadata: Option<&serde_json::Value>,
    ) -> Result<AddEvent, Em0Error> {
        let meta = metadata.cloned().unwrap_or(serde_json::json!({}));
        let conn = self.db.lock().unwrap();

        let best: Option<(i64, String, f64)> = {
            let mut stmt = conn.prepare(
                "SELECT m.rowid, m.id, v.distance
                 FROM memories_vec v JOIN memories m ON m.rowid = v.rowid
                 WHERE v.embedding MATCH ?1 AND k = 1
                   AND v.user_id = ?2 AND v.session_id = ?3 AND v.agent_id = ?4
                 ORDER BY v.distance",
            )?;
            let mut rows = stmt.query(params![
                f32_blob(emb),
                scope.user_id,
                scope.session_id,
                scope.agent_id
            ])?;
            match rows.next()? {
                Some(r) => Some((r.get(0)?, r.get(1)?, r.get(2)?)),
                None => None,
            }
        };

        if let Some((rowid, id, dist)) = best {
            let cos = dist_to_cos(dist);
            if cos > self.opts.dedup_noop {
                conn.execute(
                    "INSERT INTO history(memory_id, event, old_content, new_content, ts)
                     VALUES (?1, 'NOOP', NULL, ?2, ?3)",
                    params![id, content, now()],
                )?;
                return Ok(AddEvent::Noop { id });
            }
            if cos > self.opts.dedup_update {
                let old: String = conn.query_row(
                    "SELECT content FROM memories WHERE rowid = ?1",
                    params![rowid],
                    |r| r.get(0),
                )?;
                conn.execute(
                    "UPDATE memories SET content = ?2, updated_at = ?3 WHERE rowid = ?1",
                    params![rowid, content, now()],
                )?;
                // vec0 rows are delete+reinsert (no UPDATE on virtual tables)
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?1", params![rowid])?;
                conn.execute(
                    "INSERT INTO memories_vec(rowid, embedding, user_id, session_id, agent_id)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        rowid,
                        f32_blob(emb),
                        scope.user_id,
                        scope.session_id,
                        scope.agent_id
                    ],
                )?;
                conn.execute(
                    "INSERT INTO history(memory_id, event, old_content, new_content, ts)
                     VALUES (?1, 'UPDATE', ?2, ?3, ?4)",
                    params![id, old, content, now()],
                )?;
                return Ok(AddEvent::Update { id });
            }
        }

        let id = Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO memories(id, content, user_id, session_id, agent_id, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            params![id, content, scope.user_id, scope.session_id, scope.agent_id, meta.to_string(), now()],
        )?;
        let rowid = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO memories_vec(rowid, embedding, user_id, session_id, agent_id)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                rowid,
                f32_blob(emb),
                scope.user_id,
                scope.session_id,
                scope.agent_id
            ],
        )?;
        conn.execute(
            "INSERT INTO history(memory_id, event, old_content, new_content, ts)
             VALUES (?1, 'ADD', NULL, ?2, ?3)",
            params![id, content, now()],
        )?;
        Ok(AddEvent::Add { id })
    }

    /// Semantic (pure vector) search within a scope. Query side uses the
    /// harrier instruction prefix. `score` is cosine.
    pub fn search_vec(
        &self,
        query: &str,
        scope: &Scope,
        k: usize,
    ) -> Result<Vec<SearchHit>, Em0Error> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let q = self.h.embed_query(query).map_err(Em0Error::Harrier)?;
        let overfetch = (k.saturating_mul(4)).max(1) as i64;
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.metadata, v.distance
             FROM memories_vec v JOIN memories m ON m.rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
               AND v.user_id = ?3 AND v.session_id = ?4 AND v.agent_id = ?5
             ORDER BY v.distance
             LIMIT ?6",
        )?;
        let hits = stmt
            .query_map(
                params![
                    f32_blob(&q),
                    overfetch,
                    scope.user_id,
                    scope.session_id,
                    scope.agent_id,
                    k as i64
                ],
                |r| {
                    Ok(SearchHit {
                        id: r.get(0)?,
                        content: r.get(1)?,
                        score: dist_to_cos(r.get(3)?),
                        cosine: Some(dist_to_cos(r.get(3)?)),
                        metadata: serde_json::from_str(&r.get::<_, String>(2)?)
                            .unwrap_or(serde_json::json!({})),
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(hits)
    }

    /// Hybrid search: vector KNN + FTS5 trigram BM25, fused with Reciprocal
    /// Rank Fusion (1/(60+rank)). `score` is the RRF score; `cosine` carries
    /// the vector-path similarity when present. Queries shorter than 3
    /// characters fall back to pure vector (trigram minimum).
    pub fn search(&self, query: &str, scope: &Scope, k: usize) -> Result<Vec<SearchHit>, Em0Error> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let conn = self.db.lock().unwrap();

        // vector leg (overfetch)
        let q = self.h.embed_query(query).map_err(Em0Error::Harrier)?;
        let over = (k.saturating_mul(4)).max(1) as i64;
        let vec_leg: Vec<(i64, String, String, serde_json::Value, f32)> = {
            let mut stmt = conn.prepare(
                "SELECT m.rowid, m.id, m.content, v.distance
                 FROM memories_vec v JOIN memories m ON m.rowid = v.rowid
                 WHERE v.embedding MATCH ?1 AND k = ?2
                   AND v.user_id = ?3 AND v.session_id = ?4 AND v.agent_id = ?5
                 ORDER BY v.distance",
            )?;
            stmt.query_map(
                params![
                    f32_blob(&q),
                    over,
                    scope.user_id,
                    scope.session_id,
                    scope.agent_id
                ],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        dist_to_cos(r.get(3)?),
                    ))
                },
            )?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|(rowid, id, content, cos)| {
                let meta = serde_json::json!({});
                (rowid, id, content, meta, cos)
            })
            .collect()
        };

        let char_count = query.chars().count();
        if char_count < 3 {
            // trigram tokenizer cannot match; pure vector result
            let mut hits: Vec<SearchHit> = vec_leg
                .into_iter()
                .take(k)
                .map(|(rowid, id, content, meta, cos)| {
                    let _ = rowid;
                    SearchHit {
                        id,
                        content,
                        score: cos,
                        cosine: Some(cos),
                        metadata: meta,
                    }
                })
                .collect();
            hits.truncate(k);
            return Ok(hits);
        }

        // fts leg: phrase-quoted to dodge fts5 query syntax in user text
        let phrase = format!("\"{}\"", query.replace('"', "\"\""));
        let fts_leg: Vec<(i64, String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT m.rowid, m.id, m.content
                 FROM memories_fts f JOIN memories m ON m.rowid = f.rowid
                 WHERE memories_fts MATCH ?1
                   AND m.user_id = ?2 AND m.session_id = ?3 AND m.agent_id = ?4
                 ORDER BY rank
                 LIMIT ?5",
            )?;
            stmt.query_map(
                params![
                    phrase,
                    scope.user_id,
                    scope.session_id,
                    scope.agent_id,
                    over
                ],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?
            .collect::<Result<Vec<_>, _>>()?
        };

        // RRF fusion
        let mut rrf: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
        let mut info: std::collections::HashMap<i64, (String, String, Option<f32>)> =
            std::collections::HashMap::new();
        for (rank, (rowid, id, content, _, cos)) in vec_leg.iter().enumerate() {
            *rrf.entry(*rowid).or_insert(0.0) += 1.0 / (60.0 + rank as f32);
            info.entry(*rowid)
                .or_insert_with(|| (id.clone(), content.clone(), Some(*cos)));
        }
        for (rank, (rowid, id, content)) in fts_leg.iter().enumerate() {
            *rrf.entry(*rowid).or_insert(0.0) += 1.0 / (60.0 + rank as f32);
            info.entry(*rowid)
                .or_insert_with(|| (id.clone(), content.clone(), None));
        }
        let mut merged: Vec<(i64, f32)> = rrf.into_iter().collect();
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        merged.truncate(k);
        if merged.is_empty() {
            return Ok(Vec::new());
        }
        // one final fetch for the winners (id/content/metadata are live data)
        let rowids: Vec<String> = merged.iter().map(|(r, _)| r.to_string()).collect();
        let mut stmt = conn.prepare(&format!(
            "SELECT rowid, id, content, metadata FROM memories WHERE rowid IN ({})",
            rowids.join(",")
        ))?;
        let mut live: std::collections::HashMap<i64, (String, String, serde_json::Value)> = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or(serde_json::json!({})),
                ))
            })?
            .map(Result::unwrap)
            .map(|(a, b, c, d)| (a, (b, c, d)))
            .collect();
        Ok(merged
            .into_iter()
            .map(|(rowid, score)| {
                let (_, _, cos) = info.remove(&rowid).unwrap();
                let (id, content, metadata) = live
                    .remove(&rowid)
                    .unwrap_or_else(|| (String::new(), String::new(), serde_json::json!({})));
                SearchHit {
                    id,
                    content,
                    score,
                    cosine: cos,
                    metadata,
                }
            })
            .collect())
    }
}

impl Em0 {
    /// Overwrite a memory's content by id (re-embeds; keeps id and scope).
    /// Returns false if the id does not exist.
    pub fn update(&self, id: &str, content: &str) -> Result<bool, Em0Error> {
        let emb = self.h.embed(&[content]).map_err(Em0Error::Harrier)?[0];
        let conn = self.db.lock().unwrap();
        let found: Option<(i64, String, String, String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT rowid, content, user_id, session_id, agent_id FROM memories WHERE id = ?1",
            )?;
            let mut rows = stmt.query(params![id])?;
            match rows.next()? {
                Some(r) => Some((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                None => None,
            }
        };
        let Some((rowid, old, uid, sid, aid)) = found else {
            return Ok(false);
        };
        conn.execute(
            "UPDATE memories SET content = ?2, updated_at = ?3 WHERE rowid = ?1",
            params![rowid, content, now()],
        )?;
        conn.execute("DELETE FROM memories_vec WHERE rowid = ?1", params![rowid])?;
        conn.execute(
            "INSERT INTO memories_vec(rowid, embedding, user_id, session_id, agent_id)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![rowid, f32_blob(&emb), uid, sid, aid],
        )?;
        conn.execute(
            "INSERT INTO history(memory_id, event, old_content, new_content, ts)
             VALUES (?1, 'UPDATE', ?2, ?3, ?4)",
            params![id, old, content, now()],
        )?;
        Ok(true)
    }

    /// Delete a memory (row + vector; history is retained — it is the audit
    /// trail). Returns false if the id does not exist.
    pub fn delete(&self, id: &str) -> Result<bool, Em0Error> {
        let conn = self.db.lock().unwrap();
        let found: Option<(i64, String)> = {
            let mut stmt = conn.prepare("SELECT rowid, content FROM memories WHERE id = ?1")?;
            let mut rows = stmt.query(params![id])?;
            match rows.next()? {
                Some(r) => Some((r.get(0)?, r.get(1)?)),
                None => None,
            }
        };
        let Some((rowid, content)) = found else {
            return Ok(false);
        };
        conn.execute("DELETE FROM memories WHERE rowid = ?1", params![rowid])?;
        conn.execute("DELETE FROM memories_vec WHERE rowid = ?1", params![rowid])?;
        conn.execute(
            "INSERT INTO history(memory_id, event, old_content, new_content, ts)
             VALUES (?1, 'DELETE', ?2, NULL, ?3)",
            params![id, content, now()],
        )?;
        Ok(true)
    }

    /// List memories in a scope, newest first, paginated.
    pub fn get_all(
        &self,
        scope: &Scope,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<MemoryRecord>, Em0Error> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, content, user_id, session_id, agent_id, metadata, created_at, updated_at
             FROM memories
             WHERE user_id = ?1 AND session_id = ?2 AND agent_id = ?3
             ORDER BY created_at DESC, rowid DESC
             LIMIT ?4 OFFSET ?5",
        )?;
        let rows = stmt
            .query_map(
                params![
                    scope.user_id,
                    scope.session_id,
                    scope.agent_id,
                    limit as i64,
                    offset as i64
                ],
                |r| {
                    Ok(MemoryRecord {
                        id: r.get(0)?,
                        content: r.get(1)?,
                        scope: Scope {
                            user_id: r.get(2)?,
                            session_id: r.get(3)?,
                            agent_id: r.get(4)?,
                        },
                        metadata: serde_json::from_str(&r.get::<_, String>(5)?)
                            .unwrap_or(serde_json::json!({})),
                        created_at: r.get(6)?,
                        updated_at: r.get(7)?,
                    })
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The full audit trail for one memory (survives deletion).
    pub fn history(&self, id: &str) -> Result<Vec<HistoryEvent>, Em0Error> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT seq, memory_id, event, old_content, new_content, ts
             FROM history WHERE memory_id = ?1 ORDER BY seq",
        )?;
        let rows = stmt
            .query_map(params![id], |r| {
                Ok(HistoryEvent {
                    seq: r.get(0)?,
                    memory_id: r.get(1)?,
                    event: r.get(2)?,
                    old_content: r.get(3)?,
                    new_content: r.get(4)?,
                    ts: r.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}
