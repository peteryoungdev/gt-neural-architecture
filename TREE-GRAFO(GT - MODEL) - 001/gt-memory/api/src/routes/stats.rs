//! Statistics endpoint

use axum::{
    extract::State,
    Json,
};
use std::sync::Arc;

use crate::dto::*;
use crate::state::AppState;

/// Get system statistics
pub async fn get_stats(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<StatsResponse>> {
    let g_layer = state.memory.g_layer.read();
    let registry = state.memory.registry.read();
    let psi_share = state.psi_share.read();

    let share_stats = psi_share.get_stats();
    
    // Count total facts
    let mut total_facts = 0;
    for mu_id in registry.all_ids() {
        if let Some(mu_lock) = registry.get(&mu_id) {
            let mu = mu_lock.read();
            total_facts += mu.len();
        }
    }

    Json(ApiResponse::success(StatsResponse {
        g_layer_nodes: g_layer.node_count(),
        g_layer_edges: g_layer.edge_count(),
        memory_units: registry.len(),
        total_facts,
        psi_share_stats: PsiShareStatsResponse {
            total_mus: share_stats.total_memory_units,
            shared_mus: share_stats.shared_memory_units,
            deduplication_ratio: share_stats.deduplication_ratio,
        },
    }))
}
