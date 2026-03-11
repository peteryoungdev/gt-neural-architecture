//! Request DTOs

use serde::Deserialize;

/// Request to store a fact
#[derive(Debug, Deserialize)]
pub struct StoreFactRequest {
    /// Fact content
    pub content: String,
    
    /// Embedding vector (optional - generated if not provided)
    pub embedding: Option<Vec<f32>>,
    
    /// Associate with existing G-Layer node
    pub graph_node_id: Option<String>,
    
    /// Memory Unit name (creates new MU if not existing)
    pub memory_unit: Option<String>,
}

/// Request to query memory
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    /// Query text (used if embedding not provided)
    pub query: Option<String>,
    
    /// Query embedding
    pub embedding: Option<Vec<f32>>,
    
    /// Maximum hops in G-Layer (default: 3)
    pub max_hops: Option<usize>,
    
    /// Number of results (default: 5)
    pub top_k: Option<usize>,
    
    /// Minimum similarity threshold
    pub min_similarity: Option<f32>,
}

/// Request to add a G-Layer node
#[derive(Debug, Deserialize)]
pub struct AddNodeRequest {
    /// Node ID
    pub id: String,
    
    /// Node label
    pub label: String,
    
    /// Node embedding
    pub embedding: Option<Vec<f32>>,
    
    /// Node type/category
    pub node_type: Option<String>,
    
    /// Additional properties
    pub properties: Option<serde_json::Value>,
}

/// Request to add a G-Layer edge
#[derive(Debug, Deserialize)]
pub struct AddEdgeRequest {
    /// Subject node ID
    pub subject: String,
    
    /// Predicate (relation type)
    pub predicate: String,
    
    /// Object node ID
    pub object: String,
}

/// Request to traverse G-Layer
#[derive(Debug, Deserialize)]
pub struct TraverseRequest {
    /// Starting node IDs
    pub start_nodes: Vec<String>,
    
    /// Query embedding for relevance scoring
    pub embedding: Option<Vec<f32>>,
    
    /// Maximum hops
    pub max_hops: Option<usize>,
}

/// Request to encode text to embedding
#[derive(Debug, Deserialize)]
pub struct EncodeRequest {
    /// Text to encode
    pub text: String,
}
