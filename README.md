# em0

## MA!! 1 Remember Da $HIT

You want your agent to remember things. Simple, right? A squirrel does it offline, on a stolen acorn, with zero API keys.

So you go looking for a "memory layer" and you find THIS circus:

- **mem0** — `pip install` downloads half the known universe, then demands an **OpenAI key**. To remember. It phones Sam Altman every time your user says "hi". Your agent's brain has a monthly subscription fee now.
- **zep** — Bring-up checklist: Docker, Postgres, a graph database, and a prayer. A distributed systems cluster to recall that the user prefers dark mode.
- **letta** (MemGPT) — Runs an LLM *loop* to decide what to forget. Burning tokens to simulate dementia. Forgetting is the ONE thing humans do for free.
- **cognee** — A full ETL graph pipeline for your grocery list. Comes with networkx, vector stores, and pure unadulterated vibes.
- **evermind** — Another one. There's always another one. A new framework born every time someone discovers embeddings.

And the shared ritual, every single time: spin it up, check `htop` before storing a single bit — **1.2 GB of RSS, cold**. Deploying it means a `docker-compose.yml` longer than the Constitution, three SaaS dashboards, and an enterprise support contract. For a notepad. **A NOTEPAD WITH A MICROSERVICES DEGREE.**

**MA!! IT COSTS $15 A MONTH AND 2GB RAM TO STORE A SINGLE BOOL!!**

---

So we stopped crying in the issues tab and built a hammer:

| | them | em0 |
|---|---|---|
| **Memory Extraction** | Send telemetry + pay API fees | 2 cosine comparisons. Pure math. Offline. |
| **Deployment** | Docker Compose + 3 containers + API keys | **One static binary** |
| **State on Disk** | Postgres + Qdrant + Redis + YAML hell | **One `.db` file** |
| **Dependencies** | Half of PyPI | **Zero.** Model & tokenizer hardcoded (`#embed`) |
| **Network Traffic** | Constant | **Zero bytes.** Air-gapped by design. |

`em0` is 1,456 lines of Rust and a freestanding C kernel **IN GITHUB RELEASE**.

No cloud. No telemetry. No PyPI bloat. No microservices.

**Remembering JUST NEED a file. 🤌🤌**

---

```rust
use em0::{Em0, Scope};

let mem = Em0::open("agent.db")?;               // Or Em0::open_in_memory()
let scope = Scope::user("u-123");               // Scope defined by (user_id, session_id, agent_id)

// Write: Deterministic deduplication (no LLM required)
//   cos > 0.95 → Noop (duplicate) | cos > 0.80 → Update (rewrite, keep ID) | otherwise → Add
mem.add("The user prefers a dark theme", &scope, None)?;
mem.add_batch(&["Project codename ZF-2049 is deployed on unit 3", "today is a good day"], &scope, None)?;

// Retrieval: Hybrid search (Vector KNN + FTS5 trigram) with RRF fusion
//   Semantic queries use embeddings; specific terms/product names use exact substring matching—each acts as a fallback for the other.
let hits = mem.search("What are the user's interface preferences?", &scope, 5)?;
for h in &hits { println!("{:.4} {}", h.score, h.content); }

// Lifecycle
mem.update(&hits[0].id, "The user prefers a light theme.")?;   // Re-embeds + records in history
mem.delete(&hits[0].id)?;                                     // Deletes both the record and its vector; history is preserved for auditing
for ev in mem.history(&hits[0].id)? { /* ADD / UPDATE / DELETE ... */ }
let page: Vec<_> = mem.get_all(&scope, 20, 0)?;
```
