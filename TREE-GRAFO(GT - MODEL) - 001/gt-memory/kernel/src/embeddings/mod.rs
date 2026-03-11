//! # Embeddings Module
//!
//! Bridge for semantic embeddings and similarity operations.
//! Supports external embedding APIs (OpenAI, Cohere) and local ONNX models.

mod similarity;
mod encoder;

pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use encoder::{EmbeddingEncoder, EncoderConfig, EmbeddingProvider};

use ndarray::Array1;

/// Type alias for embedding vectors
pub type Embedding = Array1<f32>;

/// Create a zero embedding of given dimension
pub fn zero_embedding(dim: usize) -> Embedding {
    Array1::zeros(dim)
}

/// Create a random normalized embedding (for testing)
pub fn random_embedding(dim: usize) -> Embedding {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let arr = Array1::from_vec(vec);
    normalize(&arr)
}

/// Normalize an embedding to unit length
pub fn normalize(embedding: &Embedding) -> Embedding {
    let norm = embedding.dot(embedding).sqrt();
    if norm > 0.0 {
        embedding / norm
    } else {
        embedding.clone()
    }
}

/// Average multiple embeddings  
pub fn average_embeddings(embeddings: &[&Embedding]) -> Option<Embedding> {
    if embeddings.is_empty() {
        return None;
    }
    
    let dim = embeddings[0].len();
    let sum: Embedding = embeddings.iter()
        .fold(Array1::zeros(dim), |acc, e| acc + *e);
    
    Some(sum / embeddings.len() as f32)
}

/// Weighted average of embeddings
pub fn weighted_average(embeddings: &[(&Embedding, f32)]) -> Option<Embedding> {
    if embeddings.is_empty() {
        return None;
    }
    
    let dim = embeddings[0].0.len();
    let total_weight: f32 = embeddings.iter().map(|(_, w)| w).sum();
    
    if total_weight == 0.0 {
        return None;
    }
    
    let weighted_sum: Embedding = embeddings.iter().fold(
        Array1::zeros(dim),
        |acc, (e, w)| acc + (*e * *w)
    );
    
    Some(weighted_sum / total_weight)
}
