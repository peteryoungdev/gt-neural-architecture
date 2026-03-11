//! Ψ-Share Protocol - Structural sharing of Memory Units
//!
//! Implements Section 5.1: "Gestão de Redundância e O Compartilhamento Estrutural de MUs"
//!
//! The Ψ-Share protocol enables:
//! 1. Semantic similarity identification between concepts and MUs
//! 2. Shared pointer resolution for deduplication
//! 3. Subtree sharing via Ψ_offset for fine-grained reuse
//!
//! From the paper: "A eliminação da redundância estrutural via Ψ-Share não é apenas
//! uma otimização de espaço, mas uma garantia de robustez: ao corrigir um fato central
//! em uma MU compartilhada, essa correção se propaga instantaneamente para todas as
//! entidades relacionadas"

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use super::PsiPointer;
use crate::embeddings::{Embedding, cosine_similarity};
use crate::mu_tree::MUTree;

/// Registry of Memory Units for resolution
#[derive(Debug, Default)]
pub struct MURegistry {
    /// All registered Memory Units
    units: HashMap<Uuid, Arc<RwLock<MUTree>>>,
    
    /// Index of MU embeddings (Tree Encodings) for similarity search
    embeddings: HashMap<Uuid, Embedding>,
}

impl MURegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a Memory Unit
    pub fn register(&mut self, tree: MUTree) -> Uuid {
        let id = tree.id;
        let encoding = tree.get_tree_encoding();
        
        self.units.insert(id, Arc::new(RwLock::new(tree)));
        
        if let Some(emb) = encoding {
            self.embeddings.insert(id, emb);
        }
        
        id
    }

    /// Get a Memory Unit by ID
    pub fn get(&self, id: &Uuid) -> Option<Arc<RwLock<MUTree>>> {
        self.units.get(id).cloned()
    }

    /// Resolve a Ψ pointer to MU-Tree
    pub fn resolve(&self, psi: &PsiPointer) -> Option<Arc<RwLock<MUTree>>> {
        self.get(&psi.mu_address)
    }

    /// Update embedding index for an MU
    pub fn update_embedding(&mut self, id: Uuid, embedding: Embedding) {
        self.embeddings.insert(id, embedding);
    }

    /// Get count of registered MUs
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// Get all MU IDs
    pub fn all_ids(&self) -> Vec<Uuid> {
        self.units.keys().copied().collect()
    }

    /// Get MU embedding (Tree Encoding)
    pub fn get_embedding(&self, id: &Uuid) -> Option<&Embedding> {
        self.embeddings.get(id)
    }

    /// Find similar MUs by Tree Encoding
    pub fn find_similar(&self, query: &Embedding, threshold: f32) -> Vec<(Uuid, f32)> {
        let mut results: Vec<(Uuid, f32)> = self.embeddings
            .iter()
            .filter_map(|(id, emb)| {
                let sim = cosine_similarity(query, emb);
                if sim >= threshold {
                    Some((*id, sim))
                } else {
                    None
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

/// Configuration for Ψ-Share protocol
///
/// From Section 5.1: "Se o E_novo demonstra alta similaridade semântica
/// com o conteúdo de uma MU_existente (acima de um limiar τ)"
#[derive(Debug, Clone)]
pub struct PsiShareConfig {
    /// Similarity threshold τ for sharing decision
    /// Values above this trigger shared pointer creation
    pub threshold_tau: f32,
    
    /// Minimum similarity for subtree sharing
    pub subtree_threshold: f32,
    
    /// Whether to enable automatic deduplication
    pub auto_deduplicate: bool,
    
    /// Maximum depth to search for subtree sharing
    pub max_subtree_depth: usize,
}

impl Default for PsiShareConfig {
    fn default() -> Self {
        Self {
            threshold_tau: 0.85, // 85% similarity for full MU sharing
            subtree_threshold: 0.75, // 75% for subtree sharing
            auto_deduplicate: true,
            max_subtree_depth: 5,
        }
    }
}

impl PsiShareConfig {
    /// Strict config (higher thresholds, less sharing)
    pub fn strict() -> Self {
        Self {
            threshold_tau: 0.95,
            subtree_threshold: 0.90,
            auto_deduplicate: true,
            max_subtree_depth: 3,
        }
    }

    /// Aggressive config (lower thresholds, more sharing)
    pub fn aggressive() -> Self {
        Self {
            threshold_tau: 0.70,
            subtree_threshold: 0.60,
            auto_deduplicate: true,
            max_subtree_depth: 10,
        }
    }
}

/// Ψ-Share Protocol for structural deduplication
///
/// Implements Section 5.1: "O Mecanismo Ψ-Share para Deduplicação Estrutural"
///
/// The protocol:
/// 1. Identificação de Similaridade Semântica
/// 2. Resolução de Ponteiro Compartilhado  
/// 3. Compartilhamento Refinado (Sub-Árvores)
pub struct PsiShare {
    /// Configuration
    pub config: PsiShareConfig,
    
    /// Registry of Memory Units
    pub registry: MURegistry,
    
    /// Tracking of shared references: MU_ID -> [Ψ pointers using it]
    share_refs: HashMap<Uuid, Vec<ShareReference>>,
}

/// Reference tracking for shared pointers
#[derive(Debug, Clone)]
pub struct ShareReference {
    /// The Ψ pointer
    pub pointer: PsiPointer,
    
    /// ID of the concept/node using this reference
    pub owner_id: String,
    
    /// Similarity score at time of sharing
    pub similarity: f32,
}

impl PsiShare {
    /// Create new Ψ-Share protocol with default config
    pub fn new() -> Self {
        Self::with_config(PsiShareConfig::default())
    }

    /// Create with threshold τ (for backwards compatibility)
    pub fn with_threshold(threshold: f32) -> Self {
        Self::with_config(PsiShareConfig {
            threshold_tau: threshold,
            ..Default::default()
        })
    }

    /// Create with custom config
    pub fn with_config(config: PsiShareConfig) -> Self {
        Self {
            config,
            registry: MURegistry::new(),
            share_refs: HashMap::new(),
        }
    }

    /// Check if new concept should share existing MU
    ///
    /// Implements Step 1-2 from Section 5.1:
    /// "o embedding relacional e conceitual é comparado... com os embeddings
    /// de alto nível (e_Root) de MUs já existentes"
    ///
    /// Returns Some(PsiPointer) if sharing recommended (sim >= τ), None otherwise
    pub fn should_share(&self, new_embedding: &Embedding) -> Option<(PsiPointer, f32)> {
        let similar = self.registry.find_similar(new_embedding, self.config.threshold_tau);
        
        similar.first().map(|(id, sim)| {
            (PsiPointer::new(*id).create_shared(None), *sim)
        })
    }

    /// Find best sub-tree to share
    ///
    /// Implements Step 3 from Section 5.1:
    /// "se o E_novo exige apenas um subconjunto de fatos da MU_existente,
    /// o ponteiro Ψ pode incluir um Ψ_offset que direciona o acesso
    /// diretamente a uma sub-raiz específica"
    ///
    /// Returns pointer with offset if a relevant sub-tree is found
    pub fn find_shareable_subtree(
        &self,
        new_embedding: &Embedding,
    ) -> Option<(PsiPointer, f32)> {
        for (id, mu) in &self.registry.units {
            let tree = mu.read();
            
            // Check root first (full MU sharing)
            if let Some(root_emb) = tree.get_tree_encoding() {
                let sim = cosine_similarity(new_embedding, &root_emb);
                if sim >= self.config.threshold_tau {
                    return Some((PsiPointer::new(*id).create_shared(None), sim));
                }
            }
            
            // Check subtrees for more specific sharing
            if let Some(root) = tree.root() {
                // Check left subtree
                if let Some(left) = &root.left {
                    let sim = cosine_similarity(new_embedding, &left.embedding);
                    if sim >= self.config.subtree_threshold {
                        return Some((
                            PsiPointer::new(*id).create_shared(Some("L".into())),
                            sim,
                        ));
                    }
                }
                
                // Check right subtree
                if let Some(right) = &root.right {
                    let sim = cosine_similarity(new_embedding, &right.embedding);
                    if sim >= self.config.subtree_threshold {
                        return Some((
                            PsiPointer::new(*id).create_shared(Some("R".into())),
                            sim,
                        ));
                    }
                }
            }
        }

        None
    }

    /// Register a new Memory Unit (no sharing)
    pub fn register_new(&mut self, tree: MUTree) -> PsiPointer {
        let id = self.registry.register(tree);
        PsiPointer::new(id)
    }

    /// Register or share based on similarity
    ///
    /// If similar MU exists (above threshold τ), returns shared pointer.
    /// Otherwise registers new MU and returns pointer.
    ///
    /// From Section 5.1: "ele recebe o mesmo ponteiro Ψ que os nós existentes"
    pub fn register_or_share(&mut self, tree: MUTree, owner_id: Option<&str>) -> PsiPointer {
        if !self.config.auto_deduplicate {
            return self.register_new(tree);
        }

        if let Some(encoding) = tree.get_tree_encoding() {
            // Check if we should share
            if let Some((shared_ptr, similarity)) = self.should_share(&encoding) {
                // Track the sharing reference
                self.track_share(&shared_ptr, owner_id.unwrap_or("unknown"), similarity);
                return shared_ptr;
            }

            // Check for subtree sharing
            if let Some((subtree_ptr, similarity)) = self.find_shareable_subtree(&encoding) {
                self.track_share(&subtree_ptr, owner_id.unwrap_or("unknown"), similarity);
                return subtree_ptr;
            }
        }

        // No suitable sharing candidate, register new
        self.register_new(tree)
    }

    /// Track a new sharing reference
    pub fn track_share(&mut self, psi: &PsiPointer, owner_id: &str, similarity: f32) {
        let reference = ShareReference {
            pointer: psi.clone(),
            owner_id: owner_id.to_string(),
            similarity,
        };
        
        self.share_refs
            .entry(psi.mu_address)
            .or_default()
            .push(reference);
    }

    /// Get all references sharing an MU
    pub fn get_sharers(&self, mu_id: &Uuid) -> Vec<&ShareReference> {
        self.share_refs
            .get(mu_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get sharing statistics
    ///
    /// Provides metrics on deduplication effectiveness
    pub fn get_stats(&self) -> ShareStats {
        let total_mus = self.registry.len();
        let shared_mus = self.share_refs.len();
        let total_refs: usize = self.share_refs.values().map(|v| v.len()).sum();
        
        ShareStats {
            total_memory_units: total_mus,
            shared_memory_units: shared_mus,
            total_share_references: total_refs,
            deduplication_ratio: if total_refs > 0 {
                1.0 - (total_mus as f32 / (total_mus + total_refs) as f32)
            } else {
                0.0
            },
            threshold_tau: self.config.threshold_tau,
        }
    }

    /// Notify all sharers of an update
    ///
    /// From Section 5.1: "ao corrigir um fato central em uma MU compartilhada,
    /// essa correção se propaga instantaneamente para todas as entidades relacionadas"
    pub fn notify_update(&self, mu_id: &Uuid) -> Vec<String> {
        self.share_refs
            .get(mu_id)
            .map(|refs| refs.iter().map(|r| r.owner_id.clone()).collect())
            .unwrap_or_default()
    }
}

impl Default for PsiShare {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about structural sharing
#[derive(Debug, Clone)]
pub struct ShareStats {
    pub total_memory_units: usize,
    pub shared_memory_units: usize,
    pub total_share_references: usize,
    pub deduplication_ratio: f32,
    pub threshold_tau: f32,
}

impl ShareStats {
    /// Estimate memory savings from deduplication
    pub fn estimated_savings(&self, avg_mu_size_bytes: usize) -> usize {
        self.total_share_references * avg_mu_size_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_registry() {
        let mut registry = MURegistry::new();
        let tree = MUTree::new("test".into(), 3);
        let id = registry.register(tree);
        
        assert_eq!(registry.len(), 1);
        assert!(registry.get(&id).is_some());
    }

    #[test]
    fn test_psi_share_with_threshold() {
        let mut protocol = PsiShare::with_threshold(0.9);
        
        // Register first MU
        let mut tree1 = MUTree::new("tree1".into(), 2);
        tree1.insert_semantic("Fact 1".into(), array![1.0, 0.0]);
        let ptr1 = protocol.register_new(tree1);
        
        // Check if should share (similar embedding)
        let similar_emb = array![0.95, 0.05];
        let should_share = protocol.should_share(&similar_emb);
        
        // Should share because similarity > 0.9
        assert!(should_share.is_some());
        let (ptr, sim) = should_share.unwrap();
        assert!(sim >= 0.9);
    }

    #[test]
    fn test_register_or_share() {
        let mut protocol = PsiShare::with_threshold(0.8);
        
        // Register first MU
        let mut tree1 = MUTree::new("tree1".into(), 3);
        tree1.insert_semantic("Paris is capital of France".into(), array![0.9, 0.1, 0.0]);
        let ptr1 = protocol.register_or_share(tree1, Some("france"));
        
        assert!(!ptr1.is_shared);
        assert_eq!(protocol.registry.len(), 1);
        
        // Try to register similar MU (should share)
        let mut tree2 = MUTree::new("tree2".into(), 3);
        tree2.insert_semantic("Lyon is in France".into(), array![0.85, 0.15, 0.0]);
        let ptr2 = protocol.register_or_share(tree2, Some("lyon"));
        
        // Should share with first MU (similarity ~ 0.99)
        assert!(ptr2.is_shared);
        // Still only 1 MU registered
        assert_eq!(protocol.registry.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut protocol = PsiShare::new();
        let stats = protocol.get_stats();
        
        assert_eq!(stats.total_memory_units, 0);
        assert_eq!(stats.deduplication_ratio, 0.0);
    }
}
