//! Health check endpoint

use axum::Json;
use std::sync::Arc;

use crate::dto::{ApiResponse, HealthResponse};
use crate::state::AppState;

/// Health check handler
pub async fn health_check(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Json<ApiResponse<HealthResponse>> {
    Json(ApiResponse::success(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        embedding_dim: state.embedding_dim(),
    }))
}
