//! # MU-Tree Module
//!
//! Memory Unit Tree - Hierarchical factual storage with O(log n) semantic search.
//!
//! The MU-Tree organizes factual knowledge in a binary tree structure where:
//! - **Root nodes** contain high-level abstractions (Tree Encoding e_Root)
//! - **Leaf nodes** contain raw factual content (episodic memory)
//! - **Insertion** is guided by semantic similarity of embeddings

mod node;
mod tree;
mod search;

pub use node::MUNode;
pub use tree::MUTree;
pub use search::{SearchResult, SemanticSearch};
