//! # Query Module
//!
//! Dual retrieval engine for G-T Architecture.
//! Phase 1: G-Layer traversal for context
//! Phase 2: MU-Tree search for facts

mod dual_retrieval;
mod result;

pub use dual_retrieval::{DualRetrieval, RetrievalConfig};
pub use result::{QueryResult, RetrievedFact, ReasoningPath, QueryResultBuilder};

