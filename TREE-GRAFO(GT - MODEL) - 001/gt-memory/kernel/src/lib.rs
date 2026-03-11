//! # GT-Kernel
//!
//! Core implementation of the Graph-Tree (G-T) Hybrid Memory Architecture
//! for Cognitive AI Systems.
//!
//! ## Architecture Overview
//!
//! The G-T Architecture implements a dual hierarchy as specified in the paper:
//! - **G-Layer**: Macro-relational layer using Knowledge Graphs for high-level reasoning
//! - **MU-Tree Layer**: Micro-factual layer using hierarchical trees for detailed fact storage
//!
//! Integration is achieved via the Ψ (Psi) pointer, which establishes interoperability
//! by mapping graph elements to Memory Unit addresses.
//!
//! ## Key Components
//!
//! - `mu_tree`: Memory Unit Tree with O(log n) semantic search (Section 2.2)
//! - `g_layer`: Knowledge Graph with multi-hop reasoning (Section 2.1)
//! - `psi`: Ψ Protocol for symbolic resolution between layers (Section 3.1)
//! - `query`: Dual retrieval engine (Section 4.1)
//! - `sync`: Hybrid update algorithm for consistency (Section 4.2)
//! - `embeddings`: Bridge for semantic embeddings
//! - `storage`: Memory tiering and persistence (Section 5.2)
//!
//! ## References
//!
//! Based on: "Modelagem e Análise de Arquitetura de Memória Híbrida Grafo-Árvore (G-T)
//! para Sistemas Cognitivos de Inteligência Artificial" - Ricardo Juvencio Oliveira

pub mod mu_tree;
pub mod g_layer;
pub mod psi;
pub mod query;
pub mod sync;
pub mod embeddings;
pub mod storage;

// Re-exports for convenience
pub use mu_tree::{MUNode, MUTree};
pub use g_layer::{GNode, GLayer, Triple, EmbeddingWeights};
pub use psi::{PsiPointer, PsiShare, MURegistry, PsiShareConfig};
pub use query::{DualRetrieval, QueryResult, RetrievedFact, RetrievalConfig};
pub use sync::{HybridUpdate, UpdateEvent};
pub use embeddings::Embedding;
pub use storage::{MemoryTier, TierLevel, MemoryStats};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::mu_tree::{MUNode, MUTree};
    pub use crate::g_layer::{GNode, GLayer, Triple, EmbeddingWeights};
    pub use crate::psi::{PsiPointer, PsiShare, MURegistry};
    pub use crate::query::{DualRetrieval, QueryResult, RetrievedFact};
    pub use crate::sync::{HybridUpdate, UpdateEvent};
    pub use crate::embeddings::Embedding;
    pub use crate::storage::{MemoryTier, MemoryStats};
}

