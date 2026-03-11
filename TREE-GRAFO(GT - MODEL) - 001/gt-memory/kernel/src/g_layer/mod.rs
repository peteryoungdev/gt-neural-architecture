//! # G-Layer Module
//!
//! Knowledge Graph layer for macro-relational reasoning.
//! Provides multi-hop traversal and high-level context management.

mod node;
mod graph;
mod traversal;

pub use node::{GNode, EmbeddingWeights, FactualDepth};
pub use graph::{GLayer, Triple};
pub use traversal::{TraversalResult, MultiHopTraversal};

