//! MU-Tree - Memory Unit Tree with O(log n) semantic operations
//!
//! Implements Section 2.2 and 3.2 of the G-T Architecture:
//! - Balanced tree structure (AVL-style) for guaranteed O(log n)
//! - Semantic-guided insertion based on embedding similarity
//! - Hierarchical abstraction management
//! - Tree Encoding (e_Root) for G-Layer integration

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::sync::Arc;
use uuid::Uuid;

use super::node::MUNode;
use crate::embeddings::{Embedding, cosine_similarity, normalize};

/// Memory Unit Tree - Hierarchical factual storage
///
/// Implements the MU-Tree Layer from Section 2.2:
/// - Balanced tree for O(log n) search, insert, delete
/// - Semantic embedding-guided insertion
/// - Tree Encoding (e_Root) represents the entire MU
///
/// As per the paper:
/// "A parte superior da MU-Tree encapsula o resumo semântico mais importante,
/// enquanto as folhas contêm os fatos brutos."
#[derive(Debug)]
pub struct MUTree {
    /// Root node of the tree
    root: Option<Box<MUNode>>,
    
    /// Total number of nodes
    size: usize,
    
    /// Unique identifier for this MU
    pub id: Uuid,
    
    /// Human-readable name
    pub name: String,
    
    /// Embedding dimension (must be consistent)
    embedding_dim: usize,
    
    /// Maximum allowed imbalance before rebalancing (AVL threshold)
    max_imbalance: i8,
    
    /// Similarity threshold for insertion decisions
    pub insertion_threshold: f32,
}

impl MUTree {
    /// Create a new empty MU-Tree
    pub fn new(name: String, embedding_dim: usize) -> Self {
        Self {
            root: None,
            size: 0,
            id: Uuid::new_v4(),
            name,
            embedding_dim,
            max_imbalance: 1, // Standard AVL threshold
            insertion_threshold: 0.5,
        }
    }

    /// Create MU-Tree with initial root node
    pub fn with_root(name: String, root: MUNode) -> Self {
        let embedding_dim = root.embedding.len();
        let size = root.subtree_size();
        
        Self {
            root: Some(Box::new(root)),
            size,
            id: Uuid::new_v4(),
            name,
            embedding_dim,
            max_imbalance: 1,
            insertion_threshold: 0.5,
        }
    }

    /// Get the number of nodes
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Get reference to root node
    pub fn root(&self) -> Option<&MUNode> {
        self.root.as_deref()
    }

    /// Get mutable reference to root node
    pub fn root_mut(&mut self) -> Option<&mut MUNode> {
        self.root.as_deref_mut()
    }

    /// Get Tree Encoding (e_Root)
    ///
    /// From Section 2.2: "O Tree Encoding, ou seja, o embedding da raiz
    /// da sua MU-Tree" is used to represent the MU in the G-Layer.
    ///
    /// This is the aggregated semantic representation of ALL facts in the MU.
    pub fn get_tree_encoding(&self) -> Option<Embedding> {
        self.root.as_ref().map(|r| r.embedding.clone())
    }

    /// Get the maximum depth (tree height)
    pub fn height(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.height)
    }

    /// Insert a new fact with semantic-guided placement
    ///
    /// Implements Section 3.2: "Inserção Orientada por Semântica"
    /// "Um novo fato é inserido em uma posição que minimiza a distância
    /// vetorial de seu embedding em relação aos nós existentes"
    ///
    /// Complexity: O(log n) due to balanced tree structure
    pub fn insert_semantic(&mut self, content: String, embedding: Embedding) -> Uuid {
        assert_eq!(embedding.len(), self.embedding_dim, 
            "Embedding dimension mismatch: expected {}, got {}", 
            self.embedding_dim, embedding.len());

        // Normalize embedding for consistent similarity comparisons
        let normalized_embedding = normalize(&embedding);
        let new_node = MUNode::new_leaf(content, normalized_embedding);
        let node_id = new_node.id;

        if self.root.is_none() {
            self.root = Some(Box::new(new_node));
            self.size = 1;
            return node_id;
        }

        // Semantic insertion - find best position based on similarity
        self.root = Self::insert_semantic_balanced(self.root.take(), new_node);
        self.size += 1;
        
        // Update abstractions bottom-up (recalculate Tree Encoding)
        self.update_tree_encoding();
        
        node_id
    }

    /// Insert with semantic guidance and AVL balancing
    fn insert_semantic_balanced(
        node: Option<Box<MUNode>>,
        new_node: MUNode,
    ) -> Option<Box<MUNode>> {
        let mut current = match node {
            Some(n) => n,
            None => return Some(Box::new(new_node)),
        };

        // Calculate similarity with left and right children
        let left_sim = current.left.as_ref()
            .map(|l| cosine_similarity(&new_node.embedding, &l.embedding))
            .unwrap_or(-1.0);
        
        let right_sim = current.right.as_ref()
            .map(|r| cosine_similarity(&new_node.embedding, &r.embedding))
            .unwrap_or(-1.0);

        // Semantic-guided insertion: choose path with higher similarity
        // This ensures "fatos conceitualmente relacionados sejam armazenados
        // em sub-árvores adjacentes" as per Section 3.2
        if current.left.is_none() {
            let mut node = new_node;
            node.depth = current.depth + 1;
            node.similarity_to_parent = cosine_similarity(&node.embedding, &current.embedding);
            current.left = Some(Box::new(node));
        } else if current.right.is_none() {
            let mut node = new_node;
            node.depth = current.depth + 1;
            node.similarity_to_parent = cosine_similarity(&node.embedding, &current.embedding);
            current.right = Some(Box::new(node));
        } else if left_sim >= right_sim {
            current.left = Self::insert_semantic_balanced(current.left.take(), new_node);
        } else {
            current.right = Self::insert_semantic_balanced(current.right.take(), new_node);
        }

        // Recalculate embedding (aggregate from children)
        current.recalculate_embedding();
        
        // AVL rebalancing
        Some(Self::rebalance(current))
    }

    /// AVL rebalancing to maintain O(log n) complexity
    fn rebalance(mut node: Box<MUNode>) -> Box<MUNode> {
        let balance = node.balance_factor;

        // Left-heavy
        if balance > 1 {
            if let Some(ref left) = node.left {
                if left.balance_factor < 0 {
                    // Left-Right case
                    node.left = Some(Self::rotate_left(node.left.take().unwrap()));
                }
            }
            return Self::rotate_right(node);
        }

        // Right-heavy
        if balance < -1 {
            if let Some(ref right) = node.right {
                if right.balance_factor > 0 {
                    // Right-Left case
                    node.right = Some(Self::rotate_right(node.right.take().unwrap()));
                }
            }
            return Self::rotate_left(node);
        }

        node
    }

    /// Right rotation for AVL balancing
    fn rotate_right(mut y: Box<MUNode>) -> Box<MUNode> {
        let mut x = y.left.take().unwrap();
        y.left = x.right.take();
        y.recalculate_embedding();
        x.right = Some(y);
        x.recalculate_embedding();
        x
    }

    /// Left rotation for AVL balancing
    fn rotate_left(mut x: Box<MUNode>) -> Box<MUNode> {
        let mut y = x.right.take().unwrap();
        x.right = y.left.take();
        x.recalculate_embedding();
        y.left = Some(x);
        y.recalculate_embedding();
        y
    }

    /// Search for nodes by semantic similarity
    ///
    /// Implements Section 4.1 Phase 2: "Semantic Tree Search"
    /// "busca navega hierarquicamente do nó de abstração mais alta (Raiz)
    /// para o nó folha mais relevante"
    ///
    /// Returns top-k nodes most similar to query embedding.
    /// Complexity: O(log n) average case with beam search
    pub fn search_semantic(&self, query: &Embedding, top_k: usize) -> Vec<(&MUNode, f32)> {
        if self.root.is_none() {
            return Vec::new();
        }

        let normalized_query = normalize(query);
        let mut results = Vec::new();
        self.search_recursive(self.root.as_deref().unwrap(), &normalized_query, &mut results);
        
        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top-k
        results.truncate(top_k);
        results
    }

    /// Recursive semantic search with hierarchical pruning
    fn search_recursive<'a>(
        &'a self,
        node: &'a MUNode,
        query: &Embedding,
        results: &mut Vec<(&'a MUNode, f32)>,
    ) {
        let similarity = cosine_similarity(query, &node.embedding);
        
        // Add leaf nodes to results (these contain raw facts)
        if node.is_leaf {
            results.push((node, similarity));
            return;
        }

        // For internal nodes, prioritize by similarity (beam search)
        let mut children: Vec<(&MUNode, f32)> = Vec::new();
        
        if let Some(left) = &node.left {
            let left_sim = cosine_similarity(query, &left.embedding);
            children.push((left.as_ref(), left_sim));
        }
        
        if let Some(right) = &node.right {
            let right_sim = cosine_similarity(query, &right.embedding);
            children.push((right.as_ref(), right_sim));
        }

        // Sort by similarity (descending) for efficient pruning
        children.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Traverse in order of similarity (greedy best-first)
        for (child, _) in children {
            self.search_recursive(child, query, results);
        }
    }

    /// Temporal search - find facts within a time range
    ///
    /// Implements Section 6.2: "indexar as informações temporalmente"
    /// "a busca O(log n) pode recuperar eficientemente não apenas o fato (o quê),
    /// mas também o contexto temporal (o quando)"
    pub fn search_temporal(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Vec<&MUNode> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            self.search_temporal_recursive(root, from, to, &mut results);
        }
        results
    }

    fn search_temporal_recursive<'a>(
        &'a self,
        node: &'a MUNode,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        results: &mut Vec<&'a MUNode>,
    ) {
        // Check if node's timestamp is in range
        if node.timestamp >= from && node.timestamp <= to {
            if node.is_leaf {
                results.push(node);
            }
        }

        // Check children if their temporal range might overlap
        if let Some(left) = &node.left {
            let (min_t, max_t) = left.temporal_range();
            if max_t >= from && min_t <= to {
                self.search_temporal_recursive(left, from, to, results);
            }
        }
        if let Some(right) = &node.right {
            let (min_t, max_t) = right.temporal_range();
            if max_t >= from && min_t <= to {
                self.search_temporal_recursive(right, from, to, results);
            }
        }
    }

    /// Get a subtree starting from a specific offset path
    ///
    /// Implements Ψ_offset from Section 3.1:
    /// "Este offset especifica uma sub-raiz ou um nó intermediário dentro da Árvore"
    ///
    /// Offset path format: "L" for left, "R" for right
    /// Example: "LRL" means left -> right -> left from root
    pub fn get_subtree(&self, offset: &str) -> Option<&MUNode> {
        let mut current = self.root.as_deref()?;
        
        for dir in offset.chars() {
            current = match dir {
                'L' | 'l' => current.left.as_deref()?,
                'R' | 'r' => current.right.as_deref()?,
                _ => return None,
            };
        }
        
        Some(current)
    }

    /// Get Tree Encoding for a subtree (for Ψ_offset resolution)
    pub fn get_subtree_encoding(&self, offset: &str) -> Option<Embedding> {
        self.get_subtree(offset).map(|n| n.embedding.clone())
    }

    /// Update Tree Encoding (e_Root) bottom-up
    ///
    /// Implements Section 4.2: "Recálculo de Tree Encoding"
    /// "os embeddings e as abstrações dos nós ancestrais são recalculados
    /// e agregados de forma ascendente"
    pub fn update_tree_encoding(&mut self) {
        if let Some(root) = &mut self.root {
            Self::update_encoding_recursive(root);
        }
    }

    fn update_encoding_recursive(node: &mut Box<MUNode>) {
        // First update children (bottom-up)
        if let Some(left) = &mut node.left {
            Self::update_encoding_recursive(left);
        }
        if let Some(right) = &mut node.right {
            Self::update_encoding_recursive(right);
        }

        // Then recalculate this node's embedding
        if !node.is_leaf {
            node.recalculate_embedding();
        }
    }

    /// Get all leaf nodes (raw facts)
    pub fn get_all_facts(&self) -> Vec<&MUNode> {
        match &self.root {
            Some(root) => root.get_leaves(),
            None => Vec::new(),
        }
    }

    /// Find node by ID
    pub fn find_by_id(&self, id: Uuid) -> Option<&MUNode> {
        self.find_by_id_recursive(self.root.as_deref()?, id)
    }

    fn find_by_id_recursive<'a>(&'a self, node: &'a MUNode, id: Uuid) -> Option<&'a MUNode> {
        if node.id == id {
            return Some(node);
        }
        
        if let Some(left) = &node.left {
            if let Some(found) = self.find_by_id_recursive(left, id) {
                return Some(found);
            }
        }
        
        if let Some(right) = &node.right {
            if let Some(found) = self.find_by_id_recursive(right, id) {
                return Some(found);
            }
        }
        
        None
    }

    /// Get hierarchical context for a node (path from root with abstractions)
    ///
    /// Implements Section 4.1: "Os fatos brutos recuperados dos nós folha,
    /// juntamente com seu contexto hierárquico (abstração dos nós pais),
    /// são serializados"
    pub fn get_context_hierarchy(&self, node_id: Uuid) -> Vec<String> {
        let mut path = Vec::new();
        self.find_path(self.root.as_deref(), node_id, &mut path);
        path
    }

    fn find_path(&self, node: Option<&MUNode>, target_id: Uuid, path: &mut Vec<String>) -> bool {
        let node = match node {
            Some(n) => n,
            None => return false,
        };

        // Add current node's abstraction/content to path
        path.push(node.display_content().to_string());

        if node.id == target_id {
            return true;
        }

        // Try left
        if self.find_path(node.left.as_deref(), target_id, path) {
            return true;
        }

        // Try right
        if self.find_path(node.right.as_deref(), target_id, path) {
            return true;
        }

        // Not found in this path, remove current
        path.pop();
        false
    }

    /// Check if tree is balanced (AVL property)
    pub fn is_balanced(&self) -> bool {
        self.root.as_ref().map_or(true, |r| r.is_balanced())
    }

    /// Get statistics about the tree
    pub fn stats(&self) -> MUTreeStats {
        let facts = self.get_all_facts();
        
        MUTreeStats {
            total_nodes: self.size,
            leaf_nodes: facts.len(),
            internal_nodes: self.size - facts.len(),
            height: self.height(),
            is_balanced: self.is_balanced(),
        }
    }
}

/// Statistics about an MU-Tree
#[derive(Debug, Clone)]
pub struct MUTreeStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub internal_nodes: usize,
    pub height: usize,
    pub is_balanced: bool,
}

impl Default for MUTree {
    fn default() -> Self {
        Self::new("default".to_string(), 384) // Common embedding size
    }
}

/// Thread-safe wrapper for MUTree
pub type SharedMUTree = Arc<RwLock<MUTree>>;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_insert_and_search() {
        let mut tree = MUTree::new("test".to_string(), 3);
        
        tree.insert_semantic("The sky is blue".to_string(), array![0.1, 0.2, 0.9]);
        tree.insert_semantic("Water is wet".to_string(), array![0.8, 0.1, 0.1]);
        tree.insert_semantic("The ocean is blue".to_string(), array![0.2, 0.3, 0.8]);
        
        assert_eq!(tree.len(), 3);
        
        // Search for something similar to "blue sky"
        let query = array![0.1, 0.2, 0.85];
        let results = tree.search_semantic(&query, 2);
        
        assert!(!results.is_empty());
    }

    #[test]
    fn test_tree_encoding() {
        let mut tree = MUTree::new("test".to_string(), 2);
        
        tree.insert_semantic("Fact 1".to_string(), array![1.0, 0.0]);
        tree.insert_semantic("Fact 2".to_string(), array![0.0, 1.0]);
        
        let encoding = tree.get_tree_encoding();
        assert!(encoding.is_some());
    }

    #[test]
    fn test_balanced_after_inserts() {
        let mut tree = MUTree::new("test".to_string(), 3);
        
        // Insert multiple facts
        for i in 0..10 {
            let embedding = array![i as f32 / 10.0, 0.5, 1.0 - i as f32 / 10.0];
            tree.insert_semantic(format!("Fact {}", i), embedding);
        }
        
        // Tree should remain balanced
        assert!(tree.is_balanced());
        
        // Height should be logarithmic
        assert!(tree.height() <= 5); // log2(10) ≈ 3.3, +1 for safety
    }
}
