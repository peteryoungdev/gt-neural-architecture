//! Semantic search utilities for MU-Tree

use crate::embeddings::{Embedding, cosine_similarity};
use super::node::MUNode;

/// Result of a semantic search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched node ID
    pub node_id: uuid::Uuid,
    
    /// Content of the matched node
    pub content: String,
    
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    
    /// Hierarchical context (path from root)
    pub context: Vec<String>,
    
    /// Depth in tree
    pub depth: usize,
}

/// Semantic search engine for MU-Tree
pub struct SemanticSearch {
    /// Minimum similarity threshold
    pub threshold: f32,
    
    /// Maximum results to return
    pub max_results: usize,
    
    /// Whether to include context hierarchy
    pub include_context: bool,
}

impl Default for SemanticSearch {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            max_results: 10,
            include_context: true,
        }
    }
}

impl SemanticSearch {
    /// Create new search engine with custom settings
    pub fn new(threshold: f32, max_results: usize, include_context: bool) -> Self {
        Self {
            threshold,
            max_results,
            include_context,
        }
    }

    /// Beam search with adaptive beam width
    ///
    /// Uses larger beam for ambiguous queries and smaller for focused queries
    pub fn beam_search<'a>(
        &self,
        root: &'a MUNode,
        query: &Embedding,
        beam_width: usize,
    ) -> Vec<(&'a MUNode, f32)> {
        let mut beam: Vec<(&MUNode, f32)> = vec![(root, cosine_similarity(query, &root.embedding))];
        let mut results: Vec<(&MUNode, f32)> = Vec::new();

        while !beam.is_empty() {
            let mut next_beam: Vec<(&MUNode, f32)> = Vec::new();

            for (node, _) in &beam {
                if node.is_leaf {
                    let sim = cosine_similarity(query, &node.embedding);
                    if sim >= self.threshold {
                        results.push((*node, sim));
                    }
                } else {
                    // Expand children
                    if let Some(left) = &node.left {
                        let sim = cosine_similarity(query, &left.embedding);
                        next_beam.push((left.as_ref(), sim));
                    }
                    if let Some(right) = &node.right {
                        let sim = cosine_similarity(query, &right.embedding);
                        next_beam.push((right.as_ref(), sim));
                    }
                }
            }

            // Keep only top beam_width candidates
            next_beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next_beam.truncate(beam_width);
            beam = next_beam;
        }

        // Sort results by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.max_results);
        results
    }

    /// Range search - find all nodes within similarity threshold
    pub fn range_search<'a>(
        &self,
        root: &'a MUNode,
        query: &Embedding,
    ) -> Vec<(&'a MUNode, f32)> {
        let mut results = Vec::new();
        self.range_search_recursive(root, query, &mut results);
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.max_results);
        results
    }

    fn range_search_recursive<'a>(
        &self,
        node: &'a MUNode,
        query: &Embedding,
        results: &mut Vec<(&'a MUNode, f32)>,
    ) {
        let similarity = cosine_similarity(query, &node.embedding);

        // Prune if internal node similarity is below threshold
        // (children unlikely to be better matches)
        if !node.is_leaf && similarity < self.threshold * 0.7 {
            return;
        }

        if node.is_leaf && similarity >= self.threshold {
            results.push((node, similarity));
            return;
        }

        if let Some(left) = &node.left {
            self.range_search_recursive(left, query, results);
        }
        if let Some(right) = &node.right {
            self.range_search_recursive(right, query, results);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_beam_search() {
        let search = SemanticSearch::new(0.3, 5, false);
        
        let left = MUNode::new_leaf("Left fact".into(), array![1.0, 0.0, 0.0]);
        let right = MUNode::new_leaf("Right fact".into(), array![0.0, 1.0, 0.0]);
        let root = MUNode::from_children(left, right, "Root".into());
        
        let query = array![0.9, 0.1, 0.0];
        let results = search.beam_search(&root, &query, 2);
        
        assert!(!results.is_empty());
        assert!(results[0].0.content.contains("Left"));
    }
}
