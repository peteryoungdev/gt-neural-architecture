//! GNode - Node in the Knowledge Graph (G-Layer)
//!
//! Implements Section 2.2 and 3.1 of the G-T Architecture:
//! - Embedding is aggregation of (1) relational position + (2) Tree Encoding
//! - Ψ pointer for symbolic link to MU-Tree
//! - Properties for additional metadata

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::embeddings::Embedding;
use crate::psi::PsiPointer;

/// Configuration for combined embedding calculation
/// 
/// From Section 2.2: The node embedding "deve ser uma agregação de dois componentes:
/// (1) O embedding de sua posição relacional (vizinhos no G-Layer) e
/// (2) O Tree Encoding, ou seja, o embedding da raiz da sua MU-Tree"
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingWeights {
    /// Weight for relational embedding (α)
    /// Paper suggests relational embedding should be primary
    pub relational: f32,
    
    /// Weight for Tree Encoding (1 - α)
    /// Factual content representation
    pub factual: f32,
}

impl Default for EmbeddingWeights {
    fn default() -> Self {
        // Default: 60% relational, 40% factual as per typical neuro-symbolic balance
        Self {
            relational: 0.6,
            factual: 0.4,
        }
    }
}

impl EmbeddingWeights {
    /// Create balanced weights (50/50)
    pub fn balanced() -> Self {
        Self { relational: 0.5, factual: 0.5 }
    }

    /// Create relational-heavy weights (70/30)
    pub fn relational_heavy() -> Self {
        Self { relational: 0.7, factual: 0.3 }
    }

    /// Create factual-heavy weights (30/70)
    pub fn factual_heavy() -> Self {
        Self { relational: 0.3, factual: 0.7 }
    }

    /// Validate weights sum to 1.0
    pub fn is_valid(&self) -> bool {
        (self.relational + self.factual - 1.0).abs() < 0.001
    }
}

/// Node in the G-Layer Knowledge Graph
///
/// From Section 3.1: "O G-Layer funciona como o índice semântico e relacional
/// do sistema de memória. Seu design deve priorizar a esparsidade e a
/// relevância conceitual, focando em conceitos de alto impacto."
///
/// Each node can contain a Ψ pointer to a Memory Unit (MU-Tree)
/// for detailed factual knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNode {
    /// Unique identifier
    pub id: String,
    
    /// Human-readable label
    pub label: String,
    
    /// Relational embedding (position in graph based on neighbors)
    /// This is component (1) of the combined embedding
    #[serde(with = "embedding_serde")]
    pub embedding: Embedding,
    
    /// Tree Encoding from associated MU-Tree (e_Root)
    /// This is component (2) of the combined embedding
    /// From Section 2.2: "o embedding da raiz da sua MU-Tree"
    #[serde(with = "option_embedding_serde")]
    pub tree_encoding: Option<Embedding>,
    
    /// Pre-computed combined embedding (cached for performance)
    /// Recalculated when relational embedding or tree_encoding changes
    #[serde(with = "option_embedding_serde")]
    combined_embedding_cache: Option<Embedding>,
    
    /// Weights for combining embeddings
    #[serde(skip)]
    embedding_weights: EmbeddingWeights,
    
    /// Pointer to Memory Unit (Ψ)
    /// From Section 3.1: "cada nó ou aresta pode conter um ponteiro de resolução Ψ"
    pub psi: Option<PsiPointer>,
    
    /// Additional properties (flexible key-value store)
    pub properties: HashMap<String, serde_json::Value>,
    
    /// Node type/category (for type-based indexing)
    pub node_type: Option<String>,
    
    /// Whether this node has been updated (for sync tracking)
    #[serde(skip)]
    pub is_dirty: bool,
}

impl GNode {
    /// Create a new G-Layer node
    pub fn new(id: String, label: String, embedding: Embedding) -> Self {
        Self {
            id,
            label,
            embedding,
            tree_encoding: None,
            combined_embedding_cache: None,
            embedding_weights: EmbeddingWeights::default(),
            psi: None,
            properties: HashMap::new(),
            node_type: None,
            is_dirty: false,
        }
    }

    /// Create node with custom embedding weights
    pub fn with_weights(
        id: String,
        label: String,
        embedding: Embedding,
        weights: EmbeddingWeights,
    ) -> Self {
        let mut node = Self::new(id, label, embedding);
        node.embedding_weights = weights;
        node
    }

    /// Create node with PSI pointer to MU-Tree
    pub fn with_psi(
        id: String,
        label: String,
        embedding: Embedding,
        psi: PsiPointer,
    ) -> Self {
        Self {
            id,
            label,
            embedding,
            tree_encoding: None,
            combined_embedding_cache: None,
            embedding_weights: EmbeddingWeights::default(),
            psi: Some(psi),
            properties: HashMap::new(),
            node_type: None,
            is_dirty: true,
        }
    }

    /// Set the Ψ pointer
    pub fn set_psi(&mut self, psi: PsiPointer) {
        self.psi = Some(psi);
        self.is_dirty = true;
    }

    /// Update relational embedding (from GNN message passing)
    pub fn update_relational_embedding(&mut self, embedding: Embedding) {
        self.embedding = embedding;
        self.combined_embedding_cache = None; // Invalidate cache
        self.is_dirty = true;
    }

    /// Update Tree Encoding from associated MU-Tree
    ///
    /// From Section 4.2: "Este novo embedding e_Root é propagado como uma
    /// atualização de característica (feature update) para o nó E_j
    /// correspondente no G-Layer"
    pub fn update_tree_encoding(&mut self, tree_encoding: Embedding) {
        self.tree_encoding = Some(tree_encoding);
        self.combined_embedding_cache = None; // Invalidate cache
        self.is_dirty = true;
    }

    /// Clear Tree Encoding (when MU is detached)
    pub fn clear_tree_encoding(&mut self) {
        self.tree_encoding = None;
        self.combined_embedding_cache = None;
        self.is_dirty = true;
    }

    /// Get combined embedding (relational + tree encoding)
    ///
    /// Implements Section 2.2: "A representação vetorial (embedding) do nó do
    /// Grafo deve ser uma agregação de dois componentes:
    /// (1) O embedding de sua posição relacional (vizinhos no G-Layer) e
    /// (2) O Tree Encoding, ou seja, o embedding da raiz da sua MU-Tree"
    ///
    /// This produces: combined = α * relational + (1-α) * tree_encoding
    pub fn get_combined_embedding(&self) -> Embedding {
        // Return cached if available
        if let Some(cached) = &self.combined_embedding_cache {
            return cached.clone();
        }

        match &self.tree_encoding {
            Some(te) => {
                // Weighted combination as per Section 2.2
                let alpha = self.embedding_weights.relational;
                &self.embedding * alpha + te * (1.0 - alpha)
            }
            None => {
                // No tree encoding: use pure relational embedding
                self.embedding.clone()
            }
        }
    }

    /// Compute and cache combined embedding
    pub fn compute_combined_embedding(&mut self) {
        let combined = match &self.tree_encoding {
            Some(te) => {
                let alpha = self.embedding_weights.relational;
                &self.embedding * alpha + te * (1.0 - alpha)
            }
            None => self.embedding.clone(),
        };
        self.combined_embedding_cache = Some(combined);
        self.is_dirty = false;
    }

    /// Set embedding weights
    pub fn set_embedding_weights(&mut self, weights: EmbeddingWeights) {
        self.embedding_weights = weights;
        self.combined_embedding_cache = None; // Invalidate cache
    }

    /// Get embedding weights
    pub fn get_embedding_weights(&self) -> EmbeddingWeights {
        self.embedding_weights
    }

    /// Add a property
    pub fn set_property(&mut self, key: &str, value: serde_json::Value) {
        self.properties.insert(key.to_string(), value);
    }

    /// Get a property
    pub fn get_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }

    /// Check if node has MU-Tree association
    pub fn has_memory(&self) -> bool {
        self.psi.is_some()
    }

    /// Check if node has tree encoding set
    pub fn has_tree_encoding(&self) -> bool {
        self.tree_encoding.is_some()
    }

    /// Get the factual "depth" of this node
    /// From Section 1.4: "O G-Layer trata da relação, e a MU-Tree lida
    /// com a profundidade do conhecimento"
    pub fn factual_depth(&self) -> FactualDepth {
        match (&self.psi, &self.tree_encoding) {
            (Some(_), Some(_)) => FactualDepth::Full, // Has MU and encoding
            (Some(_), None) => FactualDepth::Pending, // Has MU, needs sync
            (None, _) => FactualDepth::None, // Pure relational node
        }
    }
}

/// Level of factual knowledge associated with a node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactualDepth {
    /// No associated MU-Tree
    None,
    /// Has MU-Tree but Tree Encoding not yet synced
    Pending,
    /// Fully synced with Tree Encoding
    Full,
}

impl PartialEq for GNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for GNode {}

impl std::hash::Hash for GNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Serialization for Embedding
mod embedding_serde {
    use ndarray::Array1;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(embedding: &Array1<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        embedding.to_vec().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Array1<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<f32>::deserialize(deserializer)?;
        Ok(Array1::from_vec(vec))
    }
}

mod option_embedding_serde {
    use ndarray::Array1;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(embedding: &Option<Array1<f32>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        embedding.as_ref().map(|e| e.to_vec()).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Array1<f32>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<Vec<f32>>::deserialize(deserializer)?;
        Ok(opt.map(Array1::from_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_new_node() {
        let node = GNode::new(
            "node1".to_string(),
            "Test Node".to_string(),
            array![0.1, 0.2, 0.3],
        );
        
        assert_eq!(node.id, "node1");
        assert_eq!(node.label, "Test Node");
        assert!(!node.has_memory());
        assert_eq!(node.factual_depth(), FactualDepth::None);
    }

    #[test]
    fn test_combined_embedding_default_weights() {
        let mut node = GNode::new(
            "node1".to_string(),
            "Test".to_string(),
            array![1.0, 0.0],
        );
        
        // Without tree encoding
        let emb1 = node.get_combined_embedding();
        assert_eq!(emb1, array![1.0, 0.0]);
        
        // With tree encoding (default: 60% relational, 40% factual)
        node.update_tree_encoding(array![0.0, 1.0]);
        let emb2 = node.get_combined_embedding();
        
        // 0.6 * [1.0, 0.0] + 0.4 * [0.0, 1.0] = [0.6, 0.4]
        assert!((emb2[0] - 0.6).abs() < 0.001);
        assert!((emb2[1] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_combined_embedding_custom_weights() {
        let mut node = GNode::with_weights(
            "node1".to_string(),
            "Test".to_string(),
            array![1.0, 0.0],
            EmbeddingWeights::balanced(),
        );
        
        node.update_tree_encoding(array![0.0, 1.0]);
        let emb = node.get_combined_embedding();
        
        // 0.5 * [1.0, 0.0] + 0.5 * [0.0, 1.0] = [0.5, 0.5]
        assert!((emb[0] - 0.5).abs() < 0.001);
        assert!((emb[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_factual_depth() {
        let mut node = GNode::new("n1".into(), "Test".into(), array![1.0]);
        assert_eq!(node.factual_depth(), FactualDepth::None);
        
        node.set_psi(crate::psi::PsiPointer::new(uuid::Uuid::new_v4()));
        assert_eq!(node.factual_depth(), FactualDepth::Pending);
        
        node.update_tree_encoding(array![1.0]);
        assert_eq!(node.factual_depth(), FactualDepth::Full);
    }
}
