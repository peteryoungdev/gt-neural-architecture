//! Memory endpoints

use axum::{
    extract::{Path, State},
    Json,
};
use ndarray::Array1;
use std::sync::Arc;
use uuid::Uuid;

use gt_kernel::{
    mu_tree::MUTree,
    psi::PsiPointer,
    sync::UpdateEvent,
};

use crate::dto::*;
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

/// Store a new fact
pub async fn store_fact(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StoreFactRequest>,
) -> ApiResult<Json<ApiResponse<StoreFactResponse>>> {
    // Validate/get embedding
    let embedding = match req.embedding {
        Some(vec) => {
            if vec.len() != state.embedding_dim {
                return Err(ApiError::bad_request(format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    state.embedding_dim, vec.len()
                )));
            }
            Array1::from_vec(vec)
        }
        None => {
            // Use random embedding for now (in production, would call external service)
            gt_kernel::embeddings::random_embedding(state.embedding_dim)
        }
    };

    let mut psi_share = state.psi_share.write();
    let mut registry = state.memory.registry.write();
    let mut g_layer = state.memory.g_layer.write();

    // Create or get MU
    let (mu_id, fact_id, shared) = if let Some(node_id) = &req.graph_node_id {
        // Check if node has existing PSI
        if let Some(psi) = g_layer.resolve_psi(node_id) {
            // Insert into existing MU
            if let Some(mu_lock) = registry.resolve(psi) {
                let mut mu = mu_lock.write();
                let fact_id = mu.insert_semantic(req.content.clone(), embedding);
                (mu.id, fact_id, psi.is_shared)
            } else {
                return Err(ApiError::not_found("Memory unit not found"));
            }
        } else {
            // Create new MU for this node
            let mu_name = req.memory_unit.unwrap_or_else(|| format!("mu_{}", node_id));
            let mut mu = MUTree::new(mu_name, state.embedding_dim);
            let fact_id = mu.insert_semantic(req.content.clone(), embedding);
            let mu_id = mu.id;
            
            let psi = psi_share.register_or_share(mu);
            
            if let Some(node) = g_layer.get_node_mut(node_id) {
                node.set_psi(psi.clone());
            }
            
            (mu_id, fact_id, psi.is_shared)
        }
    } else {
        // Create standalone MU
        let mu_name = req.memory_unit.unwrap_or_else(|| format!("mu_{}", Uuid::new_v4()));
        let mut mu = MUTree::new(mu_name, state.embedding_dim);
        let fact_id = mu.insert_semantic(req.content.clone(), embedding);
        let mu_id = mu.id;
        
        let psi = psi_share.register_or_share(mu);
        (mu_id, fact_id, psi.is_shared)
    };

    // Trigger sync update
    drop(g_layer);
    drop(registry);
    drop(psi_share);
    
    // Would trigger hybrid update here in background
    // state.updater.process(UpdateEvent::FactInserted { ... });

    Ok(Json(ApiResponse::success(StoreFactResponse {
        fact_id,
        memory_unit_id: mu_id,
        shared,
    })))
}

/// Query memory
pub async fn query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> ApiResult<Json<ApiResponse<QueryResponse>>> {
    // Get embedding
    let embedding = match req.embedding {
        Some(vec) => {
            if vec.len() != state.embedding_dim {
                return Err(ApiError::bad_request("Embedding dimension mismatch"));
            }
            Array1::from_vec(vec)
        }
        None => {
            // In production, would encode query text
            return Err(ApiError::bad_request("Embedding required (text encoding not yet implemented)"));
        }
    };

    let g_layer = state.memory.g_layer.read();
    let registry = state.memory.registry.read();

    // Execute dual retrieval
    let result = state.retrieval.query(&embedding, &g_layer, &registry);

    // Convert to response
    let facts: Vec<FactResponse> = result.facts.iter().map(|f| FactResponse {
        id: f.id,
        content: f.content.clone(),
        similarity: f.similarity,
        source_mu: f.source_mu,
        source_node: f.source_node.clone(),
        context: f.context_hierarchy.clone(),
    }).collect();

    let reasoning_path: Vec<PathStepResponse> = result.reasoning_path.iter().map(|p| PathStepResponse {
        node_id: p.node_id.clone(),
        node_label: p.node_label.clone(),
        relation: p.relation.clone(),
        relevance: p.relevance,
    }).collect();

    Ok(Json(ApiResponse::success(QueryResponse {
        facts,
        reasoning_path,
        latency_ms: result.latency_us as f64 / 1000.0,
        nodes_activated: result.nodes_activated,
        trees_searched: result.trees_searched,
    })))
}

/// Get a specific fact
pub async fn get_fact(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<FactResponse>>> {
    let registry = state.memory.registry.read();

    // Search all MUs for the fact
    for mu_id in registry.all_ids() {
        if let Some(mu_lock) = registry.get(&mu_id) {
            let mu = mu_lock.read();
            if let Some(node) = mu.find_by_id(id) {
                let context = mu.get_context_hierarchy(id);
                
                return Ok(Json(ApiResponse::success(FactResponse {
                    id: node.id,
                    content: node.content.clone(),
                    similarity: 1.0,
                    source_mu: mu.id,
                    source_node: String::new(),
                    context,
                })));
            }
        }
    }

    Err(ApiError::not_found("Fact not found"))
}

/// Delete a fact
pub async fn delete_fact(
    State(_state): State<Arc<AppState>>,
    Path(_id): Path<Uuid>,
) -> ApiResult<Json<ApiResponse<bool>>> {
    // TODO: Implement deletion
    Err(ApiError::bad_request("Deletion not yet implemented"))
}
