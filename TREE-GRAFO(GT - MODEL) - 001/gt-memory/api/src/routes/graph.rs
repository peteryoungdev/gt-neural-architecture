//! Graph endpoints

use axum::{
    extract::{Path, State},
    Json,
};
use ndarray::Array1;
use std::sync::Arc;

use gt_kernel::g_layer::{GNode, Triple, MultiHopTraversal};

use crate::dto::*;
use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

/// Add a node to G-Layer
pub async fn add_node(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddNodeRequest>,
) -> ApiResult<Json<ApiResponse<AddNodeResponse>>> {
    let embedding = match req.embedding {
        Some(vec) => {
            if vec.len() != state.embedding_dim {
                return Err(ApiError::bad_request("Embedding dimension mismatch"));
            }
            Array1::from_vec(vec)
        }
        None => gt_kernel::embeddings::random_embedding(state.embedding_dim),
    };

    let mut g_layer = state.memory.g_layer.write();

    // Check if node already exists
    if g_layer.get_node(&req.id).is_some() {
        return Ok(Json(ApiResponse::success(AddNodeResponse {
            node_id: req.id,
            created: false,
        })));
    }

    let mut node = GNode::new(req.id.clone(), req.label, embedding);
    node.node_type = req.node_type;
    
    if let Some(props) = req.properties {
        if let Some(obj) = props.as_object() {
            for (k, v) in obj {
                node.set_property(k, v.clone());
            }
        }
    }

    g_layer.add_node(node);

    Ok(Json(ApiResponse::success(AddNodeResponse {
        node_id: req.id,
        created: true,
    })))
}

/// Add an edge to G-Layer
pub async fn add_edge(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddEdgeRequest>,
) -> ApiResult<Json<ApiResponse<AddEdgeResponse>>> {
    let mut g_layer = state.memory.g_layer.write();

    // Verify both nodes exist
    if g_layer.get_node(&req.subject).is_none() {
        return Err(ApiError::not_found(format!("Subject node '{}' not found", req.subject)));
    }
    if g_layer.get_node(&req.object).is_none() {
        return Err(ApiError::not_found(format!("Object node '{}' not found", req.object)));
    }

    g_layer.add_edge(Triple::new(
        req.subject.clone(),
        req.predicate.clone(),
        req.object.clone(),
    ));

    Ok(Json(ApiResponse::success(AddEdgeResponse {
        subject: req.subject,
        predicate: req.predicate,
        object: req.object,
        created: true,
    })))
}

/// Traverse G-Layer
pub async fn traverse(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TraverseRequest>,
) -> ApiResult<Json<ApiResponse<TraverseResponse>>> {
    let embedding = match req.embedding {
        Some(vec) => {
            if vec.len() != state.embedding_dim {
                return Err(ApiError::bad_request("Embedding dimension mismatch"));
            }
            Array1::from_vec(vec)
        }
        None => {
            return Err(ApiError::bad_request("Embedding required for traversal"));
        }
    };

    let g_layer = state.memory.g_layer.read();
    
    let traversal = MultiHopTraversal::new(
        req.max_hops.unwrap_or(3),
        5, // beam width
    );

    let start_nodes: Vec<&str> = req.start_nodes.iter().map(|s| s.as_str()).collect();
    let result = traversal.traverse(&g_layer, &start_nodes, &embedding);

    let activated_nodes: Vec<ActivatedNodeResponse> = result.activated_nodes
        .iter()
        .map(|n| ActivatedNodeResponse {
            node_id: n.node_id.clone(),
            label: n.label.clone(),
            relevance: n.relevance_score,
            hop_distance: n.hop_distance,
        })
        .collect();

    let paths: Vec<PathStepResponse> = result.reasoning_path
        .iter()
        .map(|p| PathStepResponse {
            node_id: p.to_node.clone(),
            node_label: p.to_node.clone(),
            relation: Some(p.relation.clone()),
            relevance: 0.0,
        })
        .collect();

    Ok(Json(ApiResponse::success(TraverseResponse {
        activated_nodes,
        paths,
        hops_executed: result.hops_executed,
    })))
}

/// Get a node
pub async fn get_node(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> ApiResult<Json<ApiResponse<NodeResponse>>> {
    let g_layer = state.memory.g_layer.read();

    let node = g_layer.get_node(&id)
        .ok_or_else(|| ApiError::not_found("Node not found"))?;

    Ok(Json(ApiResponse::success(NodeResponse {
        id: node.id.clone(),
        label: node.label.clone(),
        node_type: node.node_type.clone(),
        has_memory: node.has_memory(),
        properties: serde_json::to_value(&node.properties).unwrap_or_default(),
    })))
}
