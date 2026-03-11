//! # GT-API
//!
//! Universal REST API for the G-T Memory Architecture.
//! Provides LLM-agnostic endpoints for memory operations.

mod routes;
mod dto;
mod middleware;
mod error;
mod state;

use axum::{
    Router,
    routing::{get, post, delete},
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use state::AppState;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "gt_api=debug,tower_http=debug".into())
        ))
        .init();

    // Create application state
    let state = Arc::new(AppState::new());

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(routes::health::health_check))
        
        // Memory operations
        .route("/memory/store", post(routes::memory::store_fact))
        .route("/memory/query", post(routes::memory::query))
        .route("/memory/:id", get(routes::memory::get_fact))
        .route("/memory/:id", delete(routes::memory::delete_fact))
        
        // Graph operations
        .route("/graph/node", post(routes::graph::add_node))
        .route("/graph/edge", post(routes::graph::add_edge))
        .route("/graph/traverse", post(routes::graph::traverse))
        .route("/graph/node/:id", get(routes::graph::get_node))
        
        // Embedding operations
        .route("/embeddings/encode", post(routes::embeddings::encode))
        
        // Statistics
        .route("/stats", get(routes::stats::get_stats))
        
        // Add state
        .with_state(state)
        
        // Add middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        );

    // Start server
    let addr = std::env::var("GT_API_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    
    tracing::info!("🚀 GT-API starting on {}", addr);
    tracing::info!("📊 Health check: http://{}/health", addr);
    tracing::info!("📚 Store fact: POST http://{}/memory/store", addr);
    tracing::info!("🔍 Query: POST http://{}/memory/query", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
