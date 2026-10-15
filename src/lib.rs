//! em0 — local-first memory layer.
//!
//! One sqlite file is the complete memory store (see db.rs). Embedding/rerank/tokenizer all live in the
//! vendored freestanding C kernel (vendor/harrier), weights and tokenizer data
//! baked into the binary at build time (#embed). Zero runtime assets.
//!
//! The C side is single-threaded and stateless; parallelism = caller shards
//! the batch (see `embed_par`) or holds one `Harrier` per thread.

pub mod db;
pub mod harrier;
pub mod memory;
pub mod model;

pub use db::*;
pub use model::*;

#[cfg(test)]
mod test;
