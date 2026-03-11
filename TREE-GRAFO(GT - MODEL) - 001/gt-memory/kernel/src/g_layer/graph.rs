//! GLayer - Knowledge Graph implementation

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::node::GNode;
use crate::embeddings::{Embedding, cosine_similarity};
use crate::psi::PsiPointer;

/// RDF-like triple representing an edge
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Triple {
    /// Subject node ID
    pub subject: String,
    
    /// Predicate (relation type)
    pub predicate: String,
    
    /// Object node ID
    pub object: String,
}

impl Triple {
    pub fn new(subject: String, predicate: String, object: String) -> Self {
        Self { subject, predicate, object }
    }
}

/// G-Layer Knowledge Graph
///
/// Implements the macro-relational layer of the G-T Architecture.
/// Provides high-level reasoning, context management, and multi-hop inference.
#[derive(Debug)]
pub struct GLayer {
    /// All nodes in the graph
    nodes: HashMap<String, GNode>,
    
    /// All edges (triples)
    edges: Vec<Triple>,
    
    /// Adjacency list: node_id -> [(neighbor_id, predicate)]
    outgoing: HashMap<String, Vec<(String, String)>>,
    
    /// Reverse adjacency: node_id -> [(source_id, predicate)]
    incoming: HashMap<String, Vec<(String, String)>>,
    
    /// Index of nodes by type
    type_index: HashMap<String, HashSet<String>>,
}

impl GLayer {
    /// Create a new empty G-Layer
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            type_index: HashMap::new(),
        }
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GNode) {
        // Update type index
        if let Some(node_type) = &node.node_type {
            self.type_index
                .entry(node_type.clone())
                .or_default()
                .insert(node.id.clone());
        }
        
        self.nodes.insert(node.id.clone(), node);
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&GNode> {
        self.nodes.get(id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut GNode> {
        self.nodes.get_mut(id)
    }

    /// Add an edge (triple) to the graph
    pub fn add_edge(&mut self, triple: Triple) {
        // Update adjacency lists
        self.outgoing
            .entry(triple.subject.clone())
            .or_default()
            .push((triple.object.clone(), triple.predicate.clone()));
        
        self.incoming
            .entry(triple.object.clone())
            .or_default()
            .push((triple.subject.clone(), triple.predicate.clone()));
        
        self.edges.push(triple);
    }

    /// Get direct neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<(&GNode, &str)> {
        let mut neighbors = Vec::new();
        
        if let Some(out_edges) = self.outgoing.get(node_id) {
            for (neighbor_id, predicate) in out_edges {
                if let Some(node) = self.nodes.get(neighbor_id) {
                    neighbors.push((node, predicate.as_str()));
                }
            }
        }
        
        neighbors
    }

    /// Get incoming edges to a node
    pub fn get_incoming(&self, node_id: &str) -> Vec<(&GNode, &str)> {
        let mut sources = Vec::new();
        
        if let Some(in_edges) = self.incoming.get(node_id) {
            for (source_id, predicate) in in_edges {
                if let Some(node) = self.nodes.get(source_id) {
                    sources.push((node, predicate.as_str()));
                }
            }
        }
        
        sources
    }

    /// Get all neighbors within N hops
    pub fn get_n_hop_neighbors(&self, node_id: &str, hops: usize) -> Vec<&GNode> {
        let mut visited = HashSet::new();
        let mut frontier = vec![node_id.to_string()];
        
        for _ in 0..hops {
            let mut next_frontier = Vec::new();
            
            for current in &frontier {
                if visited.contains(current) {
                    continue;
                }
                visited.insert(current.clone());
                
                if let Some(out_edges) = self.outgoing.get(current) {
                    for (neighbor, _) in out_edges {
                        if !visited.contains(neighbor) {
                            next_frontier.push(neighbor.clone());
                        }
                    }
                }
            }
            
            frontier = next_frontier;
        }
        
        visited
            .iter()
            .filter(|id| *id != node_id)
            .filter_map(|id| self.nodes.get(id))
            .collect()
    }

    /// Find nodes most similar to query embedding
    pub fn find_similar(&self, query: &Embedding, top_k: usize) -> Vec<(&GNode, f32)> {
        let mut results: Vec<(&GNode, f32)> = self.nodes
            .values()
            .map(|node| {
                let combined = node.get_combined_embedding();
                let sim = cosine_similarity(query, &combined);
                (node, sim)
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Resolve PSI pointer for a node
    pub fn resolve_psi(&self, node_id: &str) -> Option<&PsiPointer> {
        self.nodes.get(node_id)?.psi.as_ref()
    }

    /// Get all nodes of a specific type
    pub fn get_by_type(&self, node_type: &str) -> Vec<&GNode> {
        self.type_index
            .get(node_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find path between two nodes (BFS)
    pub fn find_path(&self, from: &str, to: &str) -> Option<Vec<(&GNode, Option<&str>)>> {
        if from == to {
            return self.nodes.get(from).map(|n| vec![(n, None)]);
        }

        let mut visited = HashSet::new();
        let mut queue = vec![(from.to_string(), vec![(from, None::<&str>)])];
        visited.insert(from.to_string());

        while let Some((current, path)) = queue.pop() {
            if let Some(out_edges) = self.outgoing.get(&current) {
                for (neighbor, predicate) in out_edges {
                    if visited.contains(neighbor) {
                        continue;
                    }
                    
                    let mut new_path = path.clone();
                    new_path.push((neighbor.as_str(), Some(predicate.as_str())));
                    
                    if neighbor == to {
                        // Convert string refs to node refs
                        return Some(
                            new_path
                                .into_iter()
                                .filter_map(|(id, pred)| {
                                    self.nodes.get(id).map(|n| (n, pred))
                                })
                                .collect()
                        );
                    }
                    
                    visited.insert(neighbor.clone());
                    queue.push((neighbor.clone(), new_path));
                }
            }
        }

        None
    }

    /// Get all edges for a predicate type
    pub fn get_edges_by_predicate(&self, predicate: &str) -> Vec<&Triple> {
        self.edges
            .iter()
            .filter(|t| t.predicate == predicate)
            .collect()
    }

    /// Update node embedding
    pub fn update_node_embedding(&mut self, node_id: &str, embedding: Embedding) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.embedding = embedding;
        }
    }

    /// Update tree encoding for a node
    pub fn update_tree_encoding(&mut self, node_id: &str, tree_encoding: Embedding) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.update_tree_encoding(tree_encoding);
        }
    }

    /// Get all node IDs
    pub fn get_all_node_ids(&self) -> Vec<&String> {
        self.nodes.keys().collect()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let data = SerializableGLayer {
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.clone(),
        };
        serde_json::to_string_pretty(&data)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let data: SerializableGLayer = serde_json::from_str(json)?;
        let mut graph = Self::new();
        
        for node in data.nodes {
            graph.add_node(node);
        }
        
        for triple in data.edges {
            graph.add_edge(triple);
        }
        
        Ok(graph)
    }
}

impl Default for GLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper for GLayer
pub type SharedGLayer = Arc<RwLock<GLayer>>;

/// Helper struct for serialization
#[derive(Serialize, Deserialize)]
struct SerializableGLayer {
    nodes: Vec<GNode>,
    edges: Vec<Triple>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_add_and_get_node() {
        let mut graph = GLayer::new();
        let node = GNode::new("n1".into(), "Node 1".into(), array![0.1, 0.2]);
        graph.add_node(node);
        
        assert_eq!(graph.node_count(), 1);
        assert!(graph.get_node("n1").is_some());
    }

    #[test]
    fn test_add_edge_and_neighbors() {
        let mut graph = GLayer::new();
        graph.add_node(GNode::new("a".into(), "A".into(), array![1.0]));
        graph.add_node(GNode::new("b".into(), "B".into(), array![1.0]));
        graph.add_node(GNode::new("c".into(), "C".into(), array![1.0]));
        
        graph.add_edge(Triple::new("a".into(), "related_to".into(), "b".into()));
        graph.add_edge(Triple::new("a".into(), "knows".into(), "c".into()));
        
        let neighbors = graph.get_neighbors("a");
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_find_path() {
        let mut graph = GLayer::new();
        graph.add_node(GNode::new("a".into(), "A".into(), array![1.0]));
        graph.add_node(GNode::new("b".into(), "B".into(), array![1.0]));
        graph.add_node(GNode::new("c".into(), "C".into(), array![1.0]));
        
        graph.add_edge(Triple::new("a".into(), "to".into(), "b".into()));
        graph.add_edge(Triple::new("b".into(), "to".into(), "c".into()));
        
        let path = graph.find_path("a", "c");
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 3);
    }
}
