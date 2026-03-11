//! GNN-like propagation for G-Layer updates

use crate::embeddings::{Embedding, weighted_average};
use crate::g_layer::GLayer;

/// Propagator for limited message passing
///
/// Implements simplified GNN-like propagation to update
/// neighbor embeddings after a node change.
pub struct Propagator {
    /// Decay factor for propagation (0.0 to 1.0)
    pub decay: f32,
    
    /// Weight for self-embedding vs neighbor influence
    pub self_weight: f32,
}

impl Propagator {
    /// Create new propagator
    pub fn new() -> Self {
        Self {
            decay: 0.5,
            self_weight: 0.7,
        }
    }

    /// Create with custom weights
    pub fn with_weights(decay: f32, self_weight: f32) -> Self {
        Self { decay, self_weight }
    }

    /// Propagate updates from source nodes to neighbors
    ///
    /// Implements limited message passing where neighbor embeddings
    /// are slightly adjusted based on the updated source embeddings.
    pub fn propagate_update(
        &self,
        g_layer: &mut GLayer,
        source_nodes: &[String],
        hops: usize,
    ) {
        if hops == 0 {
            return;
        }

        let mut to_update: Vec<(String, Embedding)> = Vec::new();

        for source_id in source_nodes {
            if let Some(source_node) = g_layer.get_node(source_id) {
                let source_embedding = source_node.get_combined_embedding();
                
                // Get neighbors
                let neighbors = g_layer.get_neighbors(source_id);
                
                for (neighbor, _predicate) in neighbors {
                    let neighbor_embedding = neighbor.get_combined_embedding();
                    
                    // Compute updated embedding with message passing
                    let new_embedding = self.aggregate_message(
                        &neighbor_embedding,
                        &source_embedding,
                    );
                    
                    to_update.push((neighbor.id.clone(), new_embedding));
                }
            }
        }

        // Apply updates
        for (node_id, new_embedding) in to_update {
            g_layer.update_node_embedding(&node_id, new_embedding);
        }

        // Recursive propagation for remaining hops
        if hops > 1 {
            let next_sources: Vec<String> = source_nodes.to_vec();
            // Note: In a full implementation, we'd track which nodes
            // were just updated and propagate from them
            self.propagate_update(g_layer, &next_sources, hops - 1);
        }
    }

    /// Aggregate message from neighbor into node embedding
    fn aggregate_message(&self, node_emb: &Embedding, neighbor_emb: &Embedding) -> Embedding {
        // Weighted combination: node keeps most of its embedding,
        // but incorporates some influence from updated neighbor
        let neighbor_weight = (1.0 - self.self_weight) * self.decay;
        
        node_emb * self.self_weight + neighbor_emb * neighbor_weight
    }

    /// Full neighborhood aggregation (for re-encoding)
    pub fn aggregate_neighborhood(
        &self,
        g_layer: &GLayer,
        node_id: &str,
    ) -> Option<Embedding> {
        let node = g_layer.get_node(node_id)?;
        let node_emb = node.get_combined_embedding();
        
        let neighbors = g_layer.get_neighbors(node_id);
        if neighbors.is_empty() {
            return Some(node_emb);
        }

        // Collect neighbor embeddings
        let neighbor_embs: Vec<(&Embedding, f32)> = neighbors
            .iter()
            .map(|(n, _)| {
                let emb = n.get_combined_embedding();
                // For now, equal weights - could be based on edge type
                (&n.embedding, 1.0 / neighbors.len() as f32)
            })
            .collect();

        // Aggregate using weighted average
        let neighbor_avg = weighted_average(
            &neighbor_embs.iter().map(|(e, w)| (*e, *w)).collect::<Vec<_>>(),
        )?;

        // Combine with self
        Some(node_emb * self.self_weight + neighbor_avg * (1.0 - self.self_weight))
    }
}

impl Default for Propagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g_layer::{GNode, Triple};
    use ndarray::array;

    #[test]
    fn test_propagation() {
        let mut g_layer = GLayer::new();
        
        g_layer.add_node(GNode::new("a".into(), "A".into(), array![1.0, 0.0]));
        g_layer.add_node(GNode::new("b".into(), "B".into(), array![0.0, 1.0]));
        g_layer.add_edge(Triple::new("a".into(), "to".into(), "b".into()));
        
        let propagator = Propagator::new();
        propagator.propagate_update(&mut g_layer, &["a".to_string()], 1);
        
        // B's embedding should have been slightly influenced by A
        let b = g_layer.get_node("b").unwrap();
        assert!(b.embedding[0] > 0.0); // Some influence from A
    }
}
