//! Embedding encoder bridge for external services

use serde::{Deserialize, Serialize};
use thiserror::Error;
use super::Embedding;
use ndarray::Array1;

/// Errors from embedding operations
#[derive(Debug, Error)]
pub enum EncoderError {
    #[error("Embedding service unavailable: {0}")]
    ServiceUnavailable(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    
    #[error("API error: {0}")]
    ApiError(String),
}

/// Supported embedding providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    /// OpenAI text-embedding-3-small (1536 dims)
    OpenAI { api_key: String, model: String },
    
    /// Cohere embed-multilingual-v3 (1024 dims)
    Cohere { api_key: String, model: String },
    
    /// Local ONNX model
    LocalOnnx { model_path: String },
    
    /// Pass-through (embeddings provided by client)
    Passthrough { dimension: usize },
}

/// Configuration for embedding encoder
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub provider: EmbeddingProvider,
    pub normalize: bool,
    pub batch_size: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::Passthrough { dimension: 384 },
            normalize: true,
            batch_size: 32,
        }
    }
}

/// Embedding encoder - bridges external embedding services
pub struct EmbeddingEncoder {
    config: EncoderConfig,
    dimension: usize,
}

impl EmbeddingEncoder {
    /// Create new encoder with config
    pub fn new(config: EncoderConfig) -> Self {
        let dimension = match &config.provider {
            EmbeddingProvider::OpenAI { model, .. } => {
                match model.as_str() {
                    "text-embedding-3-small" => 1536,
                    "text-embedding-3-large" => 3072,
                    "text-embedding-ada-002" => 1536,
                    _ => 1536,
                }
            }
            EmbeddingProvider::Cohere { model, .. } => {
                match model.as_str() {
                    "embed-multilingual-v3.0" => 1024,
                    "embed-english-v3.0" => 1024,
                    "embed-multilingual-light-v3.0" => 384,
                    _ => 1024,
                }
            }
            EmbeddingProvider::LocalOnnx { .. } => 384, // Typical sentence-transformers
            EmbeddingProvider::Passthrough { dimension } => *dimension,
        };

        Self { config, dimension }
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Encode text to embedding (sync version for passthrough)
    pub fn encode_passthrough(&self, embedding: Vec<f32>) -> Result<Embedding, EncoderError> {
        if embedding.len() != self.dimension {
            return Err(EncoderError::DimensionMismatch {
                expected: self.dimension,
                got: embedding.len(),
            });
        }

        let mut arr = Array1::from_vec(embedding);
        
        if self.config.normalize {
            let norm = arr.dot(&arr).sqrt();
            if norm > 0.0 {
                arr /= norm;
            }
        }
        
        Ok(arr)
    }

    /// Encode batch of embeddings (passthrough)
    pub fn encode_batch_passthrough(&self, embeddings: Vec<Vec<f32>>) -> Result<Vec<Embedding>, EncoderError> {
        embeddings
            .into_iter()
            .map(|e| self.encode_passthrough(e))
            .collect()
    }

    /// Get provider info
    pub fn provider_info(&self) -> String {
        match &self.config.provider {
            EmbeddingProvider::OpenAI { model, .. } => format!("OpenAI: {}", model),
            EmbeddingProvider::Cohere { model, .. } => format!("Cohere: {}", model),
            EmbeddingProvider::LocalOnnx { model_path } => format!("LocalOnnx: {}", model_path),
            EmbeddingProvider::Passthrough { dimension } => format!("Passthrough: {}d", dimension),
        }
    }
}

impl Default for EmbeddingEncoder {
    fn default() -> Self {
        Self::new(EncoderConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_encoder() {
        let encoder = EmbeddingEncoder::default();
        assert_eq!(encoder.dimension(), 384);
        
        let embedding = vec![0.0; 384];
        let result = encoder.encode_passthrough(embedding);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dimension_mismatch() {
        let encoder = EmbeddingEncoder::default();
        let embedding = vec![0.0; 100]; // Wrong dimension
        let result = encoder.encode_passthrough(embedding);
        assert!(matches!(result, Err(EncoderError::DimensionMismatch { .. })));
    }
}
