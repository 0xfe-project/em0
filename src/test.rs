#[cfg(test)]
mod db_tests {
    use crate::db::*;
    use rusqlite::params;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// raw insert used until memory.rs lands (t8)
    fn insert_raw(e: &Em0, id: &str, content: &str, emb: &[f32]) -> i64 {
        let conn = e.db.lock().unwrap();
        conn.execute(
            "INSERT INTO memories(id, content, created_at, updated_at) VALUES (?1, ?2, ?3, ?3)",
            params![id, content, now()],
        )
        .unwrap();
        let rowid = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO memories_vec(rowid, embedding, user_id, session_id, agent_id) VALUES (?1, ?2, '', '', '')",
            params![rowid, f32_blob(emb)],
        )
        .unwrap();
        rowid
    }

    fn knn(e: &Em0, query: &[f32], k: i64) -> Vec<(i64, f32)> {
        let conn = e.db.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT rowid, distance FROM memories_vec
                 WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance",
            )
            .unwrap();
        stmt.query_map(params![f32_blob(query), k], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    const PROTEIN: &str = "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.";
    const SUMMIT: &str = "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain.";

    #[test]
    fn smoke_knn_roundtrip_and_distance_math() {
        let e = Em0::open_in_memory().unwrap();
        let embs = e.h.embed(&["hello world", PROTEIN, SUMMIT]).unwrap();
        let (hello, protein, summit) = (embs[0], embs[1], embs[2]);
        let r_hello = insert_raw(&e, "a", "hello world", &hello);
        let r_prot = insert_raw(&e, "b", PROTEIN, &protein);
        let _r_sum = insert_raw(&e, "c", SUMMIT, &summit);

        // query with the exact hello-world embedding: nearest must be itself, d ~ 0
        let hits = knn(&e, &hello, 3);
        assert_eq!(hits[0].0, r_hello);
        assert!(hits[0].1.abs() < 1e-4, "self distance {}", hits[0].1);

        // normalized vectors: L2 distance^2 = 2 - 2*cos — cross-check vs harrier cosine
        let hits_p = knn(&e, &protein, 2);
        let d_sum: f32 = hits_p.iter().find(|(r, _)| r != &r_prot).unwrap().1;
        let want = (2.0 - 2.0 * cos(&protein, &summit)).sqrt();
        assert!(
            (d_sum - want).abs() < 1e-3,
            "vec distance {d_sum} vs hand-computed {want}"
        );
    }

    #[test]
    fn smoke_file_db_and_idempotent_migration() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        {
            let e = Em0::open(path).unwrap();
            let emb = e.h.embed(&["x"]).unwrap()[0];
            insert_raw(&e, "x", "x", &emb);
        }
        // reopen: migration is IF NOT EXISTS, must not error and must see the row
        let e2 = Em0::open(path).unwrap();
        let conn = e2.db.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
        let v: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories_vec", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, 1);
    }

    #[test]
    fn smoke_vec_delete_removes_from_knn() {
        let e = Em0::open_in_memory().unwrap();
        let embs = e.h.embed(&["hello world", PROTEIN]).unwrap();
        let r1 = insert_raw(&e, "a", "hello world", &embs[0]);
        let r2 = insert_raw(&e, "b", PROTEIN, &embs[1]);
        let conn = e.db.lock().unwrap();
        conn.execute("DELETE FROM memories_vec WHERE rowid = ?1", params![r1])
            .unwrap();
        drop(conn);
        let hits = knn(&e, &embs[0], 2);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, r2);
    }
}

#[cfg(test)]
mod harrier_tests {
    use crate::harrier::*;

    macro_rules! golden_f32 {
        ($name:literal) => {{
            let bytes = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/vendor/harrier/testdata/",
                $name,
                ".f32"
            ));
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        }};
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn h() -> Harrier {
        Harrier::new().unwrap()
    }

    // ---- L0: tokenizer ----

    #[test]
    fn tok_hello_world() {
        let ids = h().encode("hello world").unwrap();
        assert_eq!(ids, vec![2, 23391, 1902, 1]);
    }

    #[test]
    fn tok_zh() {
        let ids = h().encode("今天天气不错").unwrap();
        assert_eq!(ids, vec![2, 22601, 122362, 56033, 1]); /* 今天 天气 不错 整词合并 */
    }

    #[test]
    fn tok_empty_is_bos_eos() {
        let ids = h().encode("").unwrap();
        assert_eq!(ids, vec![2, 1]);
    }

    #[test]
    fn tok_truncation_caps_and_reterminates() {
        let long = "很长的中文文本测试,".repeat(2000); // far beyond MAX_TOKENS
        let ids = h().encode(&long).unwrap();
        assert!(ids.len() <= MAX_TOKENS, "len {}", ids.len());
        assert_eq!(*ids.first().unwrap(), 2);
        assert_eq!(*ids.last().unwrap(), 1);
    }

    // ---- L1: kernel goldens (validated against llama.cpp / numpy) ----

    #[test]
    fn embed_hello_world_golden() {
        let e = h().embed(&["hello world"]).unwrap()[0];
        let g = golden_f32!("emb_hello_world");
        assert!(cos(&e, &g) >= 0.9999, "cos {}", cos(&e, &g));
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    // ---- L1.5: semantics ----

    #[test]
    fn rerank_official_example_order() {
        let h = h();
        let q = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: 什么是重排模型?";
        let d1 = "重排(Rerank)模型是在检索出候选文档后,对这些文档进行二次精细化打分和排序的模型。";
        let d3 = "微软推出的 harrier-oss-v1 是一系列支持多语言的高性能文本嵌入和重排模型。";
        let d2 = "今天天气真好,非常适合出去和朋友一起踢足球或者散步。";
        let eq = h.embed(&[q, d1, d2, d3]).unwrap();
        let (eq, e1, e2, e3) = (eq[0], eq[1], eq[2], eq[3]);
        let (s1, s2, s3) = (cos(&eq, &e1), cos(&eq, &e2), cos(&eq, &e3));
        assert!(
            s1 > s3 && s3 > s2,
            "expected d1 > d3 > d2, got {s1} > {s3} > {s2}"
        );
        assert!(s1 - s2 > 0.2, "separation too small: {s1} vs {s2}");
        // goldens
        assert!(cos(&eq, &golden_f32!("emb_rr_query")) >= 0.9999);
        assert!(cos(&e1, &golden_f32!("emb_rr_doc1")) >= 0.9999);
        assert!(cos(&e2, &golden_f32!("emb_rr_doc2")) >= 0.9999);
        assert!(cos(&e3, &golden_f32!("emb_rr_doc3")) >= 0.9999);
    }

    #[test]
    fn rerank_api_matches_manual() {
        let h = h();
        let scores = h
            .rerank(
                "什么是重排模型?",
                &["重排模型做二次打分排序。", "今天天气真好。"],
            )
            .unwrap();
        let q = h.embed_query("什么是重排模型?").unwrap();
        let ds = h
            .embed(&["重排模型做二次打分排序。", "今天天气真好。"])
            .unwrap();
        for (i, d) in ds.iter().enumerate() {
            let manual: f32 = q.iter().zip(d).map(|(x, y)| x * y).sum();
            assert!((scores[i] - manual).abs() < 1e-5);
        }
    }

    // ---- L1.6: parallel & determinism ----

    #[test]
    fn embed_par_bitwise_equals_seq() {
        let texts = [
            "hello world",
            "今天天气不错,今天天气挺好",
            "The Federal Reserve raised interest rates by 25 basis points.",
            "什么是重排模型?",
        ];
        let h = h();
        let seq = h.embed(&texts).unwrap();
        let par = h.embed_par(&texts).unwrap();
        for (a, b) in seq.iter().zip(par.iter()) {
            assert_eq!(a, b, "seq/par mismatch");
        }
    }

    #[test]
    fn deterministic_same_bits() {
        let h = h();
        let a = h.embed(&["今天天气不错"]).unwrap()[0];
        let b = h.embed(&["今天天气不错"]).unwrap()[0];
        assert_eq!(a, b);
    }

    #[test]
    fn empty_batch_ok() {
        assert!(h().embed(&[]).unwrap().is_empty());
    }

    #[test]
    fn token_count_positive() {
        assert!(h().token_count("hello world") >= 3);
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use crate::db::Em0;
    use crate::model::AddEvent;
    use crate::model::Scope;

    #[test]
    fn full_lifecycle_add_update_delete_history() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::new("u1", "sess-1", "agent-7");

        // ADD with metadata
        let id = match e
            .add(
                "我喜欢在周末去爬山。",
                &s,
                Some(&serde_json::json!({"tag": "hobby"})),
            )
            .unwrap()
        {
            AddEvent::Add { id } => id,
            other => panic!("expected Add, got {other:?}"),
        };

        // get_all sees it with metadata + scope intact
        let rows = e.get_all(&s, 10, 0).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, id);
        assert_eq!(rows[0].content, "我喜欢在周末去爬山。");
        assert_eq!(rows[0].metadata, serde_json::json!({"tag": "hobby"}));
        assert_eq!(rows[0].scope, s);
        assert!(rows[0].updated_at >= rows[0].created_at);

        // UPDATE by id: content replaced, search follows the new text
        assert!(e.update(&id, "我喜欢在周末去海边游泳。").unwrap());
        let hits = e.search("周末去海边做什么?", &s, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("海边"));
        let rows = e.get_all(&s, 10, 0).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content, "我喜欢在周末去海边游泳。");

        // history: ADD then UPDATE, in order, with contents
        let h = e.history(&id).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].event, "ADD");
        assert_eq!(h[0].new_content.as_deref(), Some("我喜欢在周末去爬山。"));
        assert_eq!(h[1].event, "UPDATE");
        assert_eq!(h[1].old_content.as_deref(), Some("我喜欢在周末去爬山。"));
        assert_eq!(
            h[1].new_content.as_deref(),
            Some("我喜欢在周末去海边游泳。")
        );
        assert!(h[1].seq > h[0].seq);

        // DELETE: gone from get_all/search/vec, history retained
        assert!(e.delete(&id).unwrap());
        assert!(e.get_all(&s, 10, 0).unwrap().is_empty());
        assert!(e.search("海边", &s, 3).unwrap().is_empty());
        {
            let conn = e.db.lock().unwrap();
            let v: i64 = conn
                .query_row("SELECT COUNT(*) FROM memories_vec", [], |r| r.get(0))
                .unwrap();
            assert_eq!(v, 0, "vec row must be cleaned");
        }
        let h = e.history(&id).unwrap();
        assert_eq!(h.len(), 3, "history survives deletion");
        assert_eq!(h[2].event, "DELETE");
        assert_eq!(
            h[2].old_content.as_deref(),
            Some("我喜欢在周末去海边游泳。")
        );

        // idempotent misses
        assert!(!e.update(&id, "x").unwrap());
        assert!(!e.delete(&id).unwrap());
    }

    #[test]
    fn get_all_pagination_and_scope() {
        let e = Em0::open_in_memory().unwrap();
        let sa = Scope::user("alice");
        let sb = Scope::user("bob");
        let texts = [
            "alice 周末喜欢去山里徒步,还带着相机拍鸟。",
            "alice 的工作是后端工程师,主要写 Rust 和数据库。",
            "alice 养了一只橘猫,名字叫元宝,特别能吃。",
        ];
        for t in texts {
            e.add(t, &sa, None).unwrap();
        }
        e.add("bob的记忆,内容完全不同。", &sb, None).unwrap();

        let p1 = e.get_all(&sa, 2, 0).unwrap();
        let p2 = e.get_all(&sa, 2, 2).unwrap();
        assert_eq!(p1.len(), 2);
        assert_eq!(p2.len(), 1);
        assert!(e.get_all(&sb, 10, 0).unwrap().len() == 1);
        // all pages distinct ids
        assert_ne!(p1[0].id, p1[1].id);
        assert_ne!(p1[0].id, p2[0].id);
    }
}

#[cfg(test)]
mod hybrid_tests {
    use crate::db::Em0;
    use crate::model::AddEvent;

    use crate::model::Scope;

    const Q_PROTEIN: &str = "how much protein should a female eat";

    #[test]
    fn exact_substring_name_is_retrievable() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::user("ops");
        e.add(
            "ZF-2049 型振动传感器安装在 3 号机组上,负责监测轴承状态。",
            &s,
            None,
        )
        .unwrap();
        e.add(
            "上月设备例行检修完成,所有温度探头和压力表校准正常。",
            &s,
            None,
        )
        .unwrap();
        // trigram leg guarantees exact-substring retrieval regardless of embeddings
        let hits = e.search("ZF-2049", &s, 2).unwrap();
        assert!(!hits.is_empty(), "hybrid must find the exact name");
        assert!(
            hits[0].content.contains("ZF-2049"),
            "top hit must be the ZF-2049 doc"
        );
    }

    #[test]
    fn semantic_query_does_not_regress_under_hybrid() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::user("u1");
        e.add("As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.", &s, None).unwrap();
        e.add("Definition of summit for English Language Learners. : 1  the highest point of a mountain.", &s, None).unwrap();
        let hits = e.search(Q_PROTEIN, &s, 2).unwrap();
        assert!(!hits.is_empty());
        assert!(
            hits[0].content.contains("protein"),
            "semantic ranking must survive RRF"
        );
    }

    #[test]
    fn fts_follows_update_and_delete() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::user("u1");
        let id = match e
            .add("部署清单里包含 Kestrel 网关和 Redis 缓存。", &s, None)
            .unwrap()
        {
            AddEvent::Add { id } => id,
            other => panic!("{other:?}"),
        };
        assert!(
            e.search("Kestrel", &s, 3).unwrap()[0]
                .content
                .contains("Kestrel")
        );
        // rewrite: new keyword must be findable, old keyword must disappear.
        // NOTE: hybrid/vec search can still surface the single remaining row for
        // ANY query (KNN has no threshold) — keyword-sync is asserted on the
        // fts leg directly.
        e.update(&id, "部署清单里包含 Nginx 网关和 Redis 缓存。")
            .unwrap();
        let hits = e.search("Nginx", &s, 3).unwrap();
        assert!(!hits.is_empty() && hits[0].content.contains("Nginx"));
        {
            let conn = e.db.lock().unwrap();
            let old_kw: i64 = conn
                .query_row(
                    r#"SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '"Kestrel"'"#,
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(old_kw, 0, "old keyword must be gone from the fts index");
            let new_kw: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '\"Nginx\"'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(new_kw, 1, "new keyword must be in the fts index");
        }
        // delete: fts emptied
        e.delete(&id).unwrap();
        {
            let conn = e.db.lock().unwrap();
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH '\"Nginx\"'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 0);
        }
    }

    #[test]
    fn short_query_falls_back_to_vector() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::user("u1");
        e.add("我爱吃苹果,尤其是红富士。", &s, None).unwrap();
        // 2 chars < trigram minimum; must not error, must still find semantically
        let hits = e.search("苹果", &s, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("苹果"));
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::db::Em0;
    use crate::model::AddEvent;

    use crate::model::Scope;

    /// End-to-end: an agent session writes mixed zh/en memories, then lives
    /// with them — retrieval, dedup, rewrite, deletion, audit.
    #[test]
    fn chinese_memory_lifecycle_end_to_end() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let e = Em0::open(tmp.path()).unwrap(); // real file db, WAL
        let session = Scope::new("user-1", "sess-42", "agent-harrier");

        // 1. batch ingest (parallel embeddings): mixed zh/en, real-world shapes
        let batch = [
            "用户偏好深色主题,界面语言是简体中文。",
            "The user's protein requirement question was answered with CDC guidance: 46 grams per day for women 19-70.",
            "Definition of summit: the highest point of a mountain, or a meeting between leaders.",
            "项目代号 ZF-2049 的振动传感器部署在 3 号机组。",
            "今天天气不错,适合出门散步。",
        ];
        let events = e.add_batch(&batch, &session, None).unwrap();
        assert_eq!(events.len(), 5);
        assert!(
            events.iter().all(|ev| matches!(ev, AddEvent::Add { .. })),
            "distinct texts must all ADD"
        );

        // 2. semantic retrieval: zh query finds zh memory, en finds en
        let zh = e.search("用户的界面偏好是什么?", &session, 2).unwrap();
        assert!(
            !zh.is_empty() && zh[0].content.contains("深色主题"),
            "zh semantic hit"
        );
        let en = e.search("how much protein per day", &session, 2).unwrap();
        assert!(en[0].content.contains("protein"), "en semantic hit");
        // exact-name retrieval via the fts leg
        let code = e.search("ZF-2049", &session, 1).unwrap();
        assert!(code[0].content.contains("ZF-2049"));

        // 3. scope privacy
        let stranger = Scope::user("stranger");
        assert!(e.search("protein", &stranger, 3).unwrap().is_empty());

        // 4. dedup on re-ingest: identical batch again -> all NOOP, counts frozen
        let again = e.add_batch(&batch, &session, None).unwrap();
        assert!(again.iter().all(|ev| matches!(ev, AddEvent::Noop { .. })));
        assert_eq!(e.get_all(&session, 100, 0).unwrap().len(), 5);

        // 5. rewrite one memory via similar phrasing (UPDATE band)
        let ev = e
            .add("今天天气挺好,适合出去走走。", &session, None)
            .unwrap();
        match ev {
            AddEvent::Update { id } => {
                let h = e.history(&id).unwrap();
                assert!(h.iter().any(|x| x.event == "UPDATE"));
            }
            other => panic!("expected Update, got {other:?}"),
        }
        assert_eq!(
            e.get_all(&session, 100, 0).unwrap().len(),
            5,
            "rewrite must not add a row"
        );

        // 6. delete one and confirm it is unretrievable by either leg
        let rows = e.get_all(&session, 100, 0).unwrap();
        let target = rows.iter().find(|r| r.content.contains("summit")).unwrap();
        e.delete(&target.id).unwrap();
        let hits = e.search("summit definition mountain", &session, 5).unwrap();
        assert!(
            hits.iter().all(|h| !h.content.contains("summit")),
            "deleted doc must not resurface"
        );
        // and reopen the file: everything survives
        drop(e);
        let e2 = Em0::open(tmp.path()).unwrap();
        let rows = e2.get_all(&session, 100, 0).unwrap();
        assert_eq!(rows.len(), 4);
        assert!(
            e2.search("深色主题", &session, 2).unwrap()[0]
                .content
                .contains("深色主题")
        );
    }
}

#[cfg(test)]
mod memory_tests {
    use crate::db::Em0;
    use crate::model::AddEvent;
    use crate::model::Em0Options;
    use crate::model::Scope;
    const PROTEIN: &str = "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon.";
    const SUMMIT: &str = "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.";
    const Q_PROTEIN: &str = "how much protein should a female eat";

    #[test]
    fn add_then_search_hits_itself_not_the_other() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::user("u1");
        e.add(PROTEIN, &s, None).unwrap();
        e.add(SUMMIT, &s, None).unwrap();
        let hits = e.search_vec(Q_PROTEIN, &s, 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].content, PROTEIN,
            "protein query must rank its own doc first"
        );
        assert!(hits[0].score > hits[1].score, "scores must descend");
        assert!(
            hits[0].score > 0.5,
            "self-hit cosine suspiciously low: {}",
            hits[0].score
        );
    }

    #[test]
    fn duplicate_add_is_noop_and_rowcount_unchanged() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::user("u1");
        let first = e.add("今天天气不错,我们出去走走吧。", &s, None).unwrap();
        let second = e.add("今天天气不错,我们出去走走吧。", &s, None).unwrap();
        assert!(matches!(first, AddEvent::Add { .. }));
        match (first, second) {
            (AddEvent::Add { id: a }, AddEvent::Noop { id: b }) => assert_eq!(a, b),
            other => panic!("expected Add then Noop, got {other:?}"),
        }
        let conn = e.db.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        let v: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories_vec", [], |r| r.get(0))
            .unwrap();
        let h: i64 = conn
            .query_row("SELECT COUNT(*) FROM history", [], |r| r.get(0))
            .unwrap();
        drop(conn);
        assert_eq!(
            (n, v, h),
            (1, 1, 2),
            "row + vec counts unchanged, two history events"
        );
    }

    #[test]
    fn rewrite_updates_in_place_same_id() {
        let e = Em0::open_in_memory().unwrap();
        let s = Scope::user("u1");
        // measured pair: cos ≈ 0.815 -> UPDATE band (0.80, 0.95]
        let a = e.add("今天天气不错", &s, None).unwrap();
        let b = e.add("今天天气挺好", &s, None).unwrap();
        let id_a = match &a {
            AddEvent::Add { id } => id.clone(),
            _ => panic!("expected first add to be Add"),
        };
        match (a, b) {
            (AddEvent::Add { .. }, AddEvent::Update { id }) => assert_eq!(id, id_a),
            other => panic!("expected Add then Update, got {other:?}"),
        }
        // content replaced, single row, and search finds the NEW text
        let hits = e.search("今天天气挺好", &s, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "今天天气挺好");
        assert_eq!(hits[0].id, id_a);
        let conn = e.db.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        drop(conn);
        assert_eq!(n, 1);
    }

    #[test]
    fn scope_isolation_no_cross_dedup() {
        let e = Em0::open_in_memory().unwrap();
        let sa = Scope::user("alice");
        let sb = Scope::user("bob");
        let a = e.add("我住在上海,在一家小公司上班。", &sa, None).unwrap();
        let b = e.add("我住在上海,在一家小公司上班。", &sb, None).unwrap();
        assert!(matches!(a, AddEvent::Add { .. }));
        assert!(
            matches!(b, AddEvent::Add { .. }),
            "dedup must not cross scopes"
        );
        // search under alice sees only her copy
        let hits = e.search("这个人住在哪里?", &sa, 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("上海"));
        let none = e
            .search("这个人住在哪里?", &Scope::user("carol"), 5)
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn thresholds_are_configurable() {
        let opts = Em0Options {
            dedup_noop: 0.99,
            dedup_update: 0.10,
        };
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let e = Em0::open_with(tmp.path(), opts).unwrap();
        let s = Scope::user("u1");
        let a = e.add(PROTEIN, &s, None).unwrap();
        // protein vs summit measures ~0.40 cosine: with update threshold 0.10
        // the second add is a rewrite of the first, not a new row
        let b = e.add(SUMMIT, &s, None).unwrap();
        match (a, b) {
            (AddEvent::Add { id: x }, AddEvent::Update { id: y }) => assert_eq!(x, y),
            other => panic!("expected Add then Update with low threshold, got {other:?}"),
        }
    }

    #[test]
    fn search_zero_k_and_empty_scope() {
        let e = Em0::open_in_memory().unwrap();
        assert!(
            e.search("anything", &crate::model::Scope::default(), 0)
                .unwrap()
                .is_empty()
        );
        let hits = e
            .search("nothing stored yet", &crate::model::Scope::default(), 3)
            .unwrap();
        assert!(hits.is_empty());
    }
}
