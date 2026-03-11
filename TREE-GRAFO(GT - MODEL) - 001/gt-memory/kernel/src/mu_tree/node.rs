//! MUNode - Node structure for Memory Unit Tree
//!
//! Based on G-T Architecture specification (Section 3.2):
//! - Content: Raw factual text (leaves) or abstraction (internal)
//! - Embedding (e_N): Semantic vector representation
//! - Abstraction: Aggregated summary of subtree
//! - Balance Factor: For AVL-style balancing to guarantee O(log n)

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::embeddings::Embedding;

/// Node in the Memory Unit Tree (MU-Tree Layer)
///
/// Implements the formal structure from Section 3.2:
/// - Upper levels: High abstraction, low detail (semantic summaries)
/// - Lower levels (leaves): Low abstraction, raw factual detail (episodic memory)
///
/// The Tree Encoding (e_Root) of the root node is used to represent
/// the entire MU in the G-Layer combined embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MUNode {
    /// Unique identifier
    pub id: Uuid,
    
    /// Node content
    /// - For leaves: Raw factual content (episodic memory)
    /// - For internal nodes: Empty (abstraction is used instead)
    pub content: String,
    
    /// Semantic embedding vector (e_N)
    /// For internal nodes: Aggregated embedding from children
    /// For leaves: Embedding of the raw content
    #[serde(with = "embedding_serde")]
    pub embedding: Embedding,
    
    /// Abstraction: Aggregated semantic summary of subtree
    /// This is the "título ou resumo" mentioned in Section 3.2
    /// Upper levels have broader abstractions, lower levels more specific
    pub abstraction: Option<String>,
    
    /// Left child (semantically "less similar" branch in BST ordering)
    pub left: Option<Box<MUNode>>,
    
    /// Right child (semantically "more similar" branch in BST ordering)
    pub right: Option<Box<MUNode>>,
    
    /// Whether this is a leaf node (contains raw factual content)
    pub is_leaf: bool,
    
    /// Timestamp for temporal indexing (Section 6.2)
    /// Enables O(log n) temporal queries: "what happened when"
    pub timestamp: DateTime<Utc>,
    
    /// Depth in tree (root = 0)
    /// Used for abstraction level management
    pub depth: usize,
    
    /// Balance factor for AVL-style balancing
    /// Ensures O(log n) complexity as per Section 2.2
    /// balance_factor = height(left) - height(right)
    pub balance_factor: i8,
    
    /// Height of subtree (for AVL balancing)
    pub height: usize,
    
    /// Semantic similarity score to parent (for insertion tracking)
    pub similarity_to_parent: f32,
}

impl MUNode {
    /// Create a new leaf node with raw factual content
    ///
    /// As per Section 3.2: Leaf nodes contain "informação de baixa abstração,
    /// detalhe factual bruto (memória episódica consolidada)"
    pub fn new_leaf(content: String, embedding: Embedding) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            embedding,
            abstraction: None,
            left: None,
            right: None,
            is_leaf: true,
            timestamp: Utc::now(),
            depth: 0,
            balance_factor: 0,
            height: 1,
            similarity_to_parent: 0.0,
        }
    }

    /// Create a new internal node with abstraction
    ///
    /// As per Section 3.2: Internal nodes contain high-level semantic
    /// summaries that encapsulate their subtree's meaning
    pub fn new_internal(abstraction: String, embedding: Embedding) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: String::new(),
            embedding,
            abstraction: Some(abstraction),
            left: None,
            right: None,
            is_leaf: false,
            timestamp: Utc::now(),
            depth: 0,
            balance_factor: 0,
            height: 1,
            similarity_to_parent: 0.0,
        }
    }

    /// Create internal node from children (bottom-up construction)
    ///
    /// Implements the aggregation described in Section 3.2:
    /// The parent's embedding is an aggregate of children's embeddings
    pub fn from_children(
        left: MUNode,
        right: MUNode,
        abstraction: String,
    ) -> Self {
        // Aggregate embeddings from children (weighted average)
        // This creates the Tree Encoding for this subtree
        let left_emb = &left.embedding;
        let right_emb = &right.embedding;
        let aggregated = (left_emb + right_emb) / 2.0;
        
        let depth = left.depth.max(right.depth) + 1;
        let left_height = left.height;
        let right_height = right.height;
        let height = left_height.max(right_height) + 1;
        let balance_factor = (left_height as i8) - (right_height as i8);

        Self {
            id: Uuid::new_v4(),
            content: String::new(),
            embedding: aggregated,
            abstraction: Some(abstraction),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            is_leaf: false,
            timestamp: Utc::now(),
            depth,
            balance_factor,
            height,
            similarity_to_parent: 0.0,
        }
    }

    /// Get the display content (abstraction for internal, content for leaf)
    pub fn display_content(&self) -> &str {
        if self.is_leaf {
            &self.content
        } else {
            self.abstraction.as_deref().unwrap_or("[No abstraction]")
        }
    }

    /// Check if node has children
    pub fn has_children(&self) -> bool {
        self.left.is_some() || self.right.is_some()
    }

    /// Count all nodes in subtree
    pub fn subtree_size(&self) -> usize {
        let left_size = self.left.as_ref().map_or(0, |n| n.subtree_size());
        let right_size = self.right.as_ref().map_or(0, |n| n.subtree_size());
        1 + left_size + right_size
    }

    /// Get all leaf nodes in subtree (factual content)
    pub fn get_leaves(&self) -> Vec<&MUNode> {
        if self.is_leaf {
            return vec![self];
        }
        
        let mut leaves = Vec::new();
        if let Some(left) = &self.left {
            leaves.extend(left.get_leaves());
        }
        if let Some(right) = &self.right {
            leaves.extend(right.get_leaves());
        }
        leaves
    }

    /// Recalculate embedding by aggregating children (bottom-up)
    ///
    /// Implements the Tree Encoding recalculation from Section 4.2:
    /// "os embeddings e as abstrações dos nós ancestrais são recalculados
    /// e agregados de forma ascendente"
    pub fn recalculate_embedding(&mut self) {
        if self.is_leaf {
            return;
        }

        let mut embeddings: Vec<&Embedding> = Vec::new();
        
        if let Some(left) = &self.left {
            embeddings.push(&left.embedding);
        }
        if let Some(right) = &self.right {
            embeddings.push(&right.embedding);
        }

        if !embeddings.is_empty() {
            let sum: Embedding = embeddings.iter()
                .fold(Array1::zeros(embeddings[0].len()), |acc, e| acc + *e);
            self.embedding = sum / embeddings.len() as f32;
        }
        
        // Update height and balance factor
        self.update_height_and_balance();
    }

    /// Update height and balance factor for AVL balancing
    fn update_height_and_balance(&mut self) {
        let left_height = self.left.as_ref().map_or(0, |n| n.height);
        let right_height = self.right.as_ref().map_or(0, |n| n.height);
        
        self.height = left_height.max(right_height) + 1;
        self.balance_factor = (left_height as i8) - (right_height as i8);
    }

    /// Check if node is balanced (AVL property: |balance_factor| <= 1)
    pub fn is_balanced(&self) -> bool {
        self.balance_factor.abs() <= 1
    }

    /// Get the abstraction level (inverse of depth from root)
    /// Higher values = more abstract, lower = more detailed
    pub fn abstraction_level(&self, max_depth: usize) -> usize {
        max_depth.saturating_sub(self.depth)
    }

    /// Get temporal range of facts in subtree
    pub fn temporal_range(&self) -> (DateTime<Utc>, DateTime<Utc>) {
        if self.is_leaf {
            return (self.timestamp, self.timestamp);
        }

        let (mut min_time, mut max_time) = (self.timestamp, self.timestamp);

        if let Some(left) = &self.left {
            let (l_min, l_max) = left.temporal_range();
            if l_min < min_time { min_time = l_min; }
            if l_max > max_time { max_time = l_max; }
        }
        if let Some(right) = &self.right {
            let (r_min, r_max) = right.temporal_range();
            if r_min < min_time { min_time = r_min; }
            if r_max > max_time { max_time = r_max; }
        }

        (min_time, max_time)
    }
}

impl PartialEq for MUNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for MUNode {}

impl std::hash::Hash for MUNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Serialization support for ndarray
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_new_leaf() {
        let embedding = array![0.1, 0.2, 0.3];
        let node = MUNode::new_leaf("Test fact".to_string(), embedding.clone());
        
        assert!(node.is_leaf);
        assert_eq!(node.content, "Test fact");
        assert!(node.abstraction.is_none());
        assert_eq!(node.height, 1);
        assert_eq!(node.balance_factor, 0);
    }

    #[test]
    fn test_from_children() {
        let left = MUNode::new_leaf("Fact 1".to_string(), array![1.0, 0.0]);
        let right = MUNode::new_leaf("Fact 2".to_string(), array![0.0, 1.0]);
        
        let parent = MUNode::from_children(left, right, "Summary".to_string());
        
        assert!(!parent.is_leaf);
        assert_eq!(parent.embedding, array![0.5, 0.5]);
        assert_eq!(parent.depth, 1);
        assert_eq!(parent.height, 2);
        assert!(parent.is_balanced());
    }

    #[test]
    fn test_subtree_size() {
        let leaf1 = MUNode::new_leaf("Fact 1".to_string(), array![1.0]);
        let leaf2 = MUNode::new_leaf("Fact 2".to_string(), array![1.0]);
        let parent = MUNode::from_children(leaf1, leaf2, "Summary".to_string());
        
        assert_eq!(parent.subtree_size(), 3);
    }
}
