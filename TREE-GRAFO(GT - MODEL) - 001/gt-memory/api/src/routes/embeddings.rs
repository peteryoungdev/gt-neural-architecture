//! Embedding endpoints

use axum::{
    extract::State,
    Json,
};
use std::sync::Arc;

use crate::dto::*;
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

/// Encode text to embedding (placeholder for external service integration)
pub async fn encode(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EncodeRequest>,
) -> ApiResult<Json<ApiResponse<EncodeResponse>>> {
    // In a full implementation, this would call an external embedding service
    // For now, we generate a deterministic embedding based on the text
    
    // Simple hash-based pseudo-embedding for demo
    let embedding = generate_pseudo_embedding(&req.text, state.embedding_dim);
    
    Ok(Json(ApiResponse::success(EncodeResponse {
        embedding: embedding.clone(),
        dimension: embedding.len(),
    })))
}

/// Generate a pseudo-embedding from text (for demo purposes)
fn generate_pseudo_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut result = vec![0.0f32; dim];
    
    // Generate hash-based values
    for (i, chunk) in text.chars().collect::<Vec<_>>().chunks(3).enumerate() {
        let mut hasher = DefaultHasher::new();
        (i, chunk).hash(&mut hasher);
        let hash = hasher.finish();
        
        let idx = (hash as usize) % dim;
        result[idx] += ((hash % 1000) as f32 / 1000.0) - 0.5;
    }
    
    // Normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut result {
            *v /= norm;
        }
    }
    
    result
}
