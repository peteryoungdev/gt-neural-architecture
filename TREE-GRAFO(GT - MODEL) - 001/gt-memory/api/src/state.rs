//! Application state

use parking_lot::RwLock;
use std::sync::Arc;

use gt_kernel::{
    g_layer::GLayer,
    psi::{MURegistry, PsiShare},
    query::DualRetrieval,
    sync::HybridUpdate,
    embeddings::{EmbeddingEncoder, EncoderConfig, EmbeddingProvider},
    storage::MemoryTier,
};

/// Shared application state
pub struct AppState {
    /// Memory tier (G-Layer + MU Registry)
    pub memory: MemoryTier,
    
    /// Ψ-Share protocol
    pub psi_share: Arc<RwLock<PsiShare>>,
    
    /// Query engine
    pub retrieval: DualRetrieval,
    
    /// Update handler
    pub updater: HybridUpdate,
    
    /// Embedding encoder
    pub encoder: EmbeddingEncoder,
    
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl AppState {
    /// Create new application state
    pub fn new() -> Self {
        // Default to passthrough mode (client provides embeddings)
        let embedding_dim = std::env::var("GT_EMBEDDING_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(384);

        let config = EncoderConfig {
            provider: EmbeddingProvider::Passthrough { dimension: embedding_dim },
            normalize: true,
            batch_size: 32,
        };

        Self {
            memory: MemoryTier::default(),
            psi_share: Arc::new(RwLock::new(PsiShare::default())),
            retrieval: DualRetrieval::default(),
            updater: HybridUpdate::default(),
            encoder: EmbeddingEncoder::new(config),
            embedding_dim,
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
