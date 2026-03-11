//! Hybrid Update Algorithm
//!
//! Maintains consistency between G-Layer embeddings and MU-Tree content.
//! When facts are updated, propagates changes through the system.

use uuid::Uuid;

use crate::embeddings::Embedding;
use crate::g_layer::GLayer;
use crate::mu_tree::MUTree;
use crate::psi::MURegistry;

use super::propagation::Propagator;

/// Event types that trigger updates
#[derive(Debug, Clone)]
pub enum UpdateEvent {
    /// New fact inserted into MU-Tree
    FactInserted {
        mu_id: Uuid,
        node_id: Uuid,
        content: String,
    },
    
    /// Fact modified in MU-Tree
    FactModified {
        mu_id: Uuid,
        node_id: Uuid,
        old_content: String,
        new_content: String,
    },
    
    /// Fact deleted from MU-Tree
    FactDeleted {
        mu_id: Uuid,
        node_id: Uuid,
    },
    
    /// G-Layer node added
    NodeAdded {
        node_id: String,
    },
    
    /// G-Layer edge added
    EdgeAdded {
        subject: String,
        predicate: String,
        object: String,
    },
}

/// Hybrid Update Algorithm
///
/// Implements the three-step update process:
/// 1. Recalculate Tree Encoding (e_Root)
/// 2. Propagate to G-Layer node embeddings
/// 3. Limited GNN message passing to neighbors
pub struct HybridUpdate {
    /// Propagator for GNN-like message passing
    propagator: Propagator,
    
    /// Number of hops for neighbor propagation
    propagation_hops: usize,
}

impl HybridUpdate {
    /// Create new hybrid update handler
    pub fn new(propagation_hops: usize) -> Self {
        Self {
            propagator: Propagator::new(),
            propagation_hops,
        }
    }

    /// Process an update event
    pub fn process(
        &self,
        event: UpdateEvent,
        g_layer: &mut GLayer,
        registry: &mut MURegistry,
    ) -> UpdateResult {
        match event {
            UpdateEvent::FactInserted { mu_id, .. } |
            UpdateEvent::FactModified { mu_id, .. } |
            UpdateEvent::FactDeleted { mu_id, .. } => {
                self.handle_mu_update(mu_id, g_layer, registry)
            }
            
            UpdateEvent::NodeAdded { node_id } => {
                self.handle_node_added(&node_id, g_layer)
            }
            
            UpdateEvent::EdgeAdded { subject, object, .. } => {
                self.handle_edge_added(&subject, &object, g_layer)
            }
        }
    }

    /// Handle MU-Tree updates
    fn handle_mu_update(
        &self,
        mu_id: Uuid,
        g_layer: &mut GLayer,
        registry: &mut MURegistry,
    ) -> UpdateResult {
        let mut affected_nodes = Vec::new();

        // Step 1: Get updated Tree Encoding
        let new_encoding = if let Some(mu_lock) = registry.get(&mu_id) {
            let mut mu = mu_lock.write();
            mu.update_tree_encoding();
            mu.get_tree_encoding()
        } else {
            return UpdateResult {
                success: false,
                affected_nodes: vec![],
                error: Some("MU not found".to_string()),
            };
        };

        let encoding = match new_encoding {
            Some(e) => e,
            None => return UpdateResult {
                success: true,
                affected_nodes: vec![],
                error: None,
            },
        };

        // Step 2: Find G-Layer nodes pointing to this MU
        // Collect node_ids first to avoid borrow conflict
        let node_ids: Vec<String> = g_layer.get_all_node_ids()
            .iter()
            .map(|s| s.to_string())
            .collect();
        
        for node_id in &node_ids {
            if let Some(psi) = g_layer.resolve_psi(node_id) {
                if psi.mu_address == mu_id {
                    affected_nodes.push(node_id.clone());
                }
            }
        }
        
        // Now update the affected nodes
        for node_id in &affected_nodes {
            g_layer.update_tree_encoding(node_id, encoding.clone());
        }

        // Step 3: Propagate to neighbors
        if !affected_nodes.is_empty() {
            self.propagator.propagate_update(
                g_layer,
                &affected_nodes,
                self.propagation_hops,
            );
        }

        // Update registry embedding index
        registry.update_embedding(mu_id, encoding);

        UpdateResult {
            success: true,
            affected_nodes,
            error: None,
        }
    }

    /// Handle new G-Layer node
    fn handle_node_added(&self, node_id: &str, g_layer: &mut GLayer) -> UpdateResult {
        // Minimal update - just trigger neighbor awareness
        self.propagator.propagate_update(g_layer, &[node_id.to_string()], 1);
        
        UpdateResult {
            success: true,
            affected_nodes: vec![node_id.to_string()],
            error: None,
        }
    }

    /// Handle new G-Layer edge
    fn handle_edge_added(
        &self,
        subject: &str,
        object: &str,
        g_layer: &mut GLayer,
    ) -> UpdateResult {
        // Propagate from both endpoints
        let affected = vec![subject.to_string(), object.to_string()];
        self.propagator.propagate_update(g_layer, &affected, 1);
        
        UpdateResult {
            success: true,
            affected_nodes: affected,
            error: None,
        }
    }
}

impl Default for HybridUpdate {
    fn default() -> Self {
        Self::new(1) // Default 1 hop propagation
    }
}

/// Result of an update operation
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub success: bool,
    pub affected_nodes: Vec<String>,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g_layer::GNode;
    use crate::psi::PsiPointer;
    use ndarray::array;

    #[test]
    fn test_hybrid_update() {
        let mut g_layer = GLayer::new();
        let mut registry = MURegistry::new();
        
        // Create and register MU
        let mut tree = MUTree::new("test".into(), 3);
        tree.insert_semantic("Initial fact".into(), array![0.5, 0.5, 0.0]);
        let tree_id = tree.id;
        registry.register(tree);
        
        // Create G-Node pointing to MU
        let mut node = GNode::new("n1".into(), "Node 1".into(), array![0.5, 0.5, 0.0]);
        node.set_psi(PsiPointer::new(tree_id));
        g_layer.add_node(node);
        
        // Process update
        let updater = HybridUpdate::default();
        let result = updater.process(
            UpdateEvent::FactInserted {
                mu_id: tree_id,
                node_id: Uuid::new_v4(),
                content: "New fact".into(),
            },
            &mut g_layer,
            &mut registry,
        );
        
        assert!(result.success);
        assert!(!result.affected_nodes.is_empty());
    }
}
