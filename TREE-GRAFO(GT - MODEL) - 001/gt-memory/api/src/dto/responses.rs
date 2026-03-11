//! Response DTOs

use serde::Serialize;
use uuid::Uuid;

/// Standard success response wrapper
#[derive(Debug, Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub success: bool,
    pub data: T,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self { success: true, data }
    }
}

/// Response for storing a fact
#[derive(Debug, Serialize)]
pub struct StoreFactResponse {
    /// Created fact ID
    pub fact_id: Uuid,
    
    /// Memory Unit ID
    pub memory_unit_id: Uuid,
    
    /// Whether PSI sharing was used
    pub shared: bool,
}

/// Response for queries
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// Retrieved facts
    pub facts: Vec<FactResponse>,
    
    /// Reasoning path through G-Layer
    pub reasoning_path: Vec<PathStepResponse>,
    
    /// Query latency in milliseconds
    pub latency_ms: f64,
    
    /// Number of nodes activated
    pub nodes_activated: usize,
    
    /// Number of MU-Trees searched
    pub trees_searched: usize,
}

/// A retrieved fact
#[derive(Debug, Serialize)]
pub struct FactResponse {
    pub id: Uuid,
    pub content: String,
    pub similarity: f32,
    pub source_mu: Uuid,
    pub source_node: String,
    pub context: Vec<String>,
}

/// A step in reasoning path
#[derive(Debug, Serialize)]
pub struct PathStepResponse {
    pub node_id: String,
    pub node_label: String,
    pub relation: Option<String>,
    pub relevance: f32,
}

/// Response for adding a node
#[derive(Debug, Serialize)]
pub struct AddNodeResponse {
    pub node_id: String,
    pub created: bool,
}

/// Response for adding an edge
#[derive(Debug, Serialize)]
pub struct AddEdgeResponse {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub created: bool,
}

/// Response for G-Layer traversal
#[derive(Debug, Serialize)]
pub struct TraverseResponse {
    pub activated_nodes: Vec<ActivatedNodeResponse>,
    pub paths: Vec<PathStepResponse>,
    pub hops_executed: usize,
}

/// An activated node
#[derive(Debug, Serialize)]
pub struct ActivatedNodeResponse {
    pub node_id: String,
    pub label: String,
    pub relevance: f32,
    pub hop_distance: usize,
}

/// Response for getting a node
#[derive(Debug, Serialize)]
pub struct NodeResponse {
    pub id: String,
    pub label: String,
    pub node_type: Option<String>,
    pub has_memory: bool,
    pub properties: serde_json::Value,
}

/// Response for encoding
#[derive(Debug, Serialize)]
pub struct EncodeResponse {
    pub embedding: Vec<f32>,
    pub dimension: usize,
}

/// Statistics response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub g_layer_nodes: usize,
    pub g_layer_edges: usize,
    pub memory_units: usize,
    pub total_facts: usize,
    pub psi_share_stats: PsiShareStatsResponse,
}

#[derive(Debug, Serialize)]
pub struct PsiShareStatsResponse {
    pub total_mus: usize,
    pub shared_mus: usize,
    pub deduplication_ratio: f32,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub embedding_dim: usize,
}
