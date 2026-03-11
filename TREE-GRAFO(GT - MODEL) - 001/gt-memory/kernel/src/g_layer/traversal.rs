//! Multi-hop traversal for reasoning over the G-Layer

use std::collections::{HashSet, BinaryHeap};
use std::cmp::Ordering;

use super::graph::GLayer;
use super::node::GNode;
use crate::embeddings::{Embedding, cosine_similarity};

/// Result of a multi-hop traversal
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Activated nodes in order of relevance
    pub activated_nodes: Vec<ActivatedNode>,
    
    /// Total hops executed
    pub hops_executed: usize,
    
    /// Reasoning path (for explainability)
    pub reasoning_path: Vec<PathStep>,
}

/// A node activated during traversal
#[derive(Debug, Clone)]
pub struct ActivatedNode {
    pub node_id: String,
    pub label: String,
    pub relevance_score: f32,
    pub hop_distance: usize,
}

/// A step in the reasoning path
#[derive(Debug, Clone)]
pub struct PathStep {
    pub from_node: String,
    pub relation: String,
    pub to_node: String,
}

/// Priority queue entry for beam search
#[derive(Debug, Clone)]
struct BeamEntry {
    node_id: String,
    score: f32,
    hop: usize,
    path: Vec<PathStep>,
}

impl PartialEq for BeamEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for BeamEntry {}

impl PartialOrd for BeamEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for BeamEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Multi-hop traversal engine for G-Layer reasoning
pub struct MultiHopTraversal {
    /// Maximum number of hops
    pub max_hops: usize,
    
    /// Beam width for best-first search
    pub beam_width: usize,
    
    /// Minimum relevance threshold
    pub relevance_threshold: f32,
    
    /// Maximum nodes to activate
    pub max_activated: usize,
}

impl Default for MultiHopTraversal {
    fn default() -> Self {
        Self {
            max_hops: 3,
            beam_width: 5,
            relevance_threshold: 0.3,
            max_activated: 10,
        }
    }
}

impl MultiHopTraversal {
    /// Create new traversal engine with custom settings
    pub fn new(max_hops: usize, beam_width: usize) -> Self {
        Self {
            max_hops,
            beam_width,
            ..Default::default()
        }
    }

    /// Execute multi-hop traversal from starting nodes
    ///
    /// Implements beam search guided by semantic similarity to query
    pub fn traverse(
        &self,
        graph: &GLayer,
        start_nodes: &[&str],
        query: &Embedding,
    ) -> TraversalResult {
        let mut visited = HashSet::new();
        let mut beam = BinaryHeap::new();
        let mut activated = Vec::new();
        let mut all_paths = Vec::new();

        // Initialize beam with starting nodes
        for start_id in start_nodes {
            if let Some(node) = graph.get_node(start_id) {
                let score = cosine_similarity(query, &node.get_combined_embedding());
                beam.push(BeamEntry {
                    node_id: start_id.to_string(),
                    score,
                    hop: 0,
                    path: Vec::new(),
                });
            }
        }

        let mut current_hop = 0;

        while !beam.is_empty() && current_hop <= self.max_hops {
            // Get top beam_width entries for this hop
            let mut current_level: Vec<BeamEntry> = Vec::new();
            while let Some(entry) = beam.pop() {
                if entry.hop == current_hop && current_level.len() < self.beam_width {
                    current_level.push(entry);
                } else if entry.hop > current_hop {
                    beam.push(entry);
                    break;
                }
            }

            for entry in current_level {
                if visited.contains(&entry.node_id) {
                    continue;
                }
                visited.insert(entry.node_id.clone());

                if let Some(node) = graph.get_node(&entry.node_id) {
                    // Add to activated if above threshold
                    if entry.score >= self.relevance_threshold {
                        activated.push(ActivatedNode {
                            node_id: entry.node_id.clone(),
                            label: node.label.clone(),
                            relevance_score: entry.score,
                            hop_distance: entry.hop,
                        });
                        all_paths.extend(entry.path.clone());
                    }

                    // Expand to neighbors if not at max hops
                    if entry.hop < self.max_hops {
                        let neighbors = graph.get_neighbors(&entry.node_id);
                        
                        for (neighbor, predicate) in neighbors {
                            if visited.contains(&neighbor.id) {
                                continue;
                            }

                            let neighbor_score = cosine_similarity(
                                query,
                                &neighbor.get_combined_embedding(),
                            );

                            // Decay score by hop distance
                            let decay = 0.9_f32.powi(entry.hop as i32 + 1);
                            let adjusted_score = neighbor_score * decay;

                            let mut new_path = entry.path.clone();
                            new_path.push(PathStep {
                                from_node: entry.node_id.clone(),
                                relation: predicate.to_string(),
                                to_node: neighbor.id.clone(),
                            });

                            beam.push(BeamEntry {
                                node_id: neighbor.id.clone(),
                                score: adjusted_score,
                                hop: entry.hop + 1,
                                path: new_path,
                            });
                        }
                    }
                }
            }

            current_hop += 1;
        }

        // Sort activated by relevance
        activated.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(Ordering::Equal)
        });
        activated.truncate(self.max_activated);

        // Deduplicate paths
        let mut seen_paths = HashSet::new();
        let unique_paths: Vec<PathStep> = all_paths
            .into_iter()
            .filter(|p| {
                let key = format!("{}-{}-{}", p.from_node, p.relation, p.to_node);
                seen_paths.insert(key)
            })
            .collect();

        TraversalResult {
            activated_nodes: activated,
            hops_executed: current_hop.min(self.max_hops),
            reasoning_path: unique_paths,
        }
    }

    /// Find nodes most relevant to query and traverse from them
    pub fn traverse_from_query(
        &self,
        graph: &GLayer,
        query: &Embedding,
        initial_k: usize,
    ) -> TraversalResult {
        // Find initial entry points
        let similar = graph.find_similar(query, initial_k);
        let start_nodes: Vec<&str> = similar.iter().map(|(n, _)| n.id.as_str()).collect();
        
        self.traverse(graph, &start_nodes, query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_multi_hop_traversal() {
        let mut graph = GLayer::new();
        
        // Create a small graph: A -> B -> C
        graph.add_node(GNode::new("a".into(), "A".into(), array![1.0, 0.0, 0.0]));
        graph.add_node(GNode::new("b".into(), "B".into(), array![0.5, 0.5, 0.0]));
        graph.add_node(GNode::new("c".into(), "C".into(), array![0.0, 0.0, 1.0]));
        
        graph.add_edge(super::super::Triple::new("a".into(), "to".into(), "b".into()));
        graph.add_edge(super::super::Triple::new("b".into(), "to".into(), "c".into()));
        
        let traversal = MultiHopTraversal::new(3, 5);
        let query = array![1.0, 0.0, 0.0]; // Similar to A
        
        let result = traversal.traverse(&graph, &["a"], &query);
        
        assert!(!result.activated_nodes.is_empty());
        assert_eq!(result.activated_nodes[0].node_id, "a");
    }
}
