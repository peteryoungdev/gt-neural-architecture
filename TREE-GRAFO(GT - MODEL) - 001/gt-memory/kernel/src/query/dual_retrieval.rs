//! Dual Retrieval Engine - Two-phase query processing
//!
//! Implements Section 4.1: "Algoritmo de Recuperação Dual (G-T Query Process)"
//! 
//! Phase 1 (Contextualização): G-Layer traversal for context and reasoning
//! Phase 2 (Foco Factual): MU-Tree search for detailed facts

use std::time::Instant;

use crate::embeddings::Embedding;
use crate::g_layer::{GLayer, MultiHopTraversal};
use crate::psi::MURegistry;

use super::result::{QueryResult, QueryResultBuilder, RetrievedFact, ReasoningPath};

/// Configuration for dual retrieval
/// 
/// Tunable parameters that affect the balance between context (G-Layer)
/// and factual detail (MU-Tree) in query results.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    // === Phase 1: G-Layer Traversal ===
    
    /// Maximum hops in G-Layer multi-hop reasoning
    /// From Section 4.1: "raciocínio multi-hop, navegando pelas relações do grafo"
    pub max_hops: usize,
    
    /// Beam width for G-Layer search
    /// Controls breadth of exploration in context phase
    pub beam_width: usize,
    
    /// Minimum relevance threshold for G-Layer nodes
    pub context_threshold: f32,
    
    /// Maximum nodes to activate in G-Layer
    pub max_activated_nodes: usize,
    
    // === Phase 2: MU-Tree Search ===
    
    /// Number of top facts to retrieve per activated MU
    pub facts_per_mu: usize,
    
    /// Minimum similarity threshold for facts
    pub fact_threshold: f32,
    
    /// Maximum total facts to return
    pub max_facts: usize,
    
    /// Whether to include hierarchical context for each fact
    /// From Section 4.1: "juntamente com seu contexto hierárquico"
    pub include_context_hierarchy: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            // Phase 1
            max_hops: 3,
            beam_width: 5,
            context_threshold: 0.3,
            max_activated_nodes: 10,
            // Phase 2
            facts_per_mu: 5,
            fact_threshold: 0.3,
            max_facts: 20,
            include_context_hierarchy: true,
        }
    }
}

impl RetrievalConfig {
    /// Config optimized for speed (fewer hops, less facts)
    pub fn fast() -> Self {
        Self {
            max_hops: 2,
            beam_width: 3,
            context_threshold: 0.4,
            max_activated_nodes: 5,
            facts_per_mu: 3,
            fact_threshold: 0.4,
            max_facts: 10,
            include_context_hierarchy: false,
        }
    }

    /// Config optimized for thoroughness (more exploration)
    pub fn thorough() -> Self {
        Self {
            max_hops: 5,
            beam_width: 10,
            context_threshold: 0.2,
            max_activated_nodes: 20,
            facts_per_mu: 10,
            fact_threshold: 0.2,
            max_facts: 50,
            include_context_hierarchy: true,
        }
    }
}

/// Dual Retrieval Engine
///
/// Implements the formal G-T Query Process from Section 4.1:
///
/// **Phase 1 - Contextualização e Raciocínio (G-Layer Traversal)**
/// 1. Query embedding initiates semantic/symbolic search in G-Layer
/// 2. Multi-hop reasoning identifies relevant entities
/// 3. Ψ pointers are resolved for activated nodes
///
/// **Phase 2 - Foco Factual e Detalhe (MU-Tree Search)**
/// 1. System accesses MUs via Ψ (including offsets)
/// 2. Semantic tree search navigates hierarchically
/// 3. Efficient O(log n) retrieval of detailed facts
/// 4. Facts serialized with hierarchical context
pub struct DualRetrieval {
    config: RetrievalConfig,
}

impl DualRetrieval {
    /// Create new retrieval engine with default config
    pub fn new() -> Self {
        Self {
            config: RetrievalConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: RetrievalConfig) -> Self {
        Self { config }
    }

    /// Execute dual retrieval query
    ///
    /// Implements the complete G-T Query Process from Section 4.1
    ///
    /// # Arguments
    /// * `query` - Query embedding (derived from input)
    /// * `g_layer` - The G-Layer knowledge graph
    /// * `registry` - MU Registry for resolving Ψ pointers
    ///
    /// # Returns
    /// QueryResult with facts, reasoning path, and performance metrics
    pub fn query(
        &self,
        query: &Embedding,
        g_layer: &GLayer,
        registry: &MURegistry,
    ) -> QueryResult {
        let start_time = Instant::now();
        let mut builder = QueryResultBuilder::new();

        // ================================================================
        // PHASE 1: Contextualização e Raciocínio (G-Layer Traversal)
        // ================================================================
        // From Section 4.1: "O query embedding é usado para iniciar uma
        // busca semântica e simbólica dentro do G-Layer"
        
        let traversal = MultiHopTraversal::new(self.config.max_hops, self.config.beam_width);
        let traversal_result = traversal.traverse_from_query(
            g_layer,
            query,
            self.config.max_activated_nodes,
        );

        // Record reasoning path (the "contextual why" from G-Layer)
        // From Section 4: "tracing... the contextual 'why' (G-Layer)"
        let mut hop_tracker = 0;
        for step in &traversal_result.reasoning_path {
            builder.add_reasoning_step(ReasoningPath {
                node_id: step.to_node.clone(),
                node_label: step.to_node.clone(),
                relation: Some(step.relation.clone()),
                relevance: 0.0,
                hop: hop_tracker,
            });
        }

        // Record activated nodes with their relevance scores
        for activated in &traversal_result.activated_nodes {
            if activated.relevance_score >= self.config.context_threshold {
                builder.add_reasoning_step(ReasoningPath {
                    node_id: activated.node_id.clone(),
                    node_label: activated.label.clone(),
                    relation: None,
                    relevance: activated.relevance_score,
                    hop: activated.hop_distance,
                });
                hop_tracker = activated.hop_distance;
            }
        }

        builder.set_nodes_activated(traversal_result.activated_nodes.len());

        // ================================================================
        // PHASE 2: Foco Factual e Detalhe (MU-Tree Search)
        // ================================================================
        // From Section 4.1: "O sistema utiliza Ψ_i para acessar a MU correspondente"
        
        let mut trees_searched = 0;

        for activated in &traversal_result.activated_nodes {
            // Skip nodes below context threshold
            if activated.relevance_score < self.config.context_threshold {
                continue;
            }

            // Resolve Ψ pointer
            // From Section 4.1 Phase 1 Step 3: "o ponteiro Ψ_i associado é recuperado"
            if let Some(psi) = g_layer.resolve_psi(&activated.node_id) {
                if let Some(mu_lock) = registry.resolve(psi) {
                    let mu = mu_lock.read();
                    trees_searched += 1;

                    // Get subtree if offset specified (Ψ_offset)
                    // From Section 3.1: "offset especifica uma sub-raiz"
                    let search_root = match &psi.offset {
                        Some(offset) => mu.get_subtree(offset),
                        None => mu.root(),
                    };

                    if search_root.is_some() {
                        // Semantic Tree Search
                        // From Section 4.1 Phase 2 Step 2: "busca navega hierarquicamente
                        // do nó de abstração mais alta (Raiz) para o nó folha mais relevante"
                        let search_results = mu.search_semantic(query, self.config.facts_per_mu);

                        for (node, similarity) in search_results {
                            // Apply fact threshold
                            if similarity >= self.config.fact_threshold {
                                // Get hierarchical context if enabled
                                // From Section 4.1: "Os fatos brutos recuperados dos nós folha,
                                // juntamente com seu contexto hierárquico (abstração dos nós pais)"
                                let context = if self.config.include_context_hierarchy {
                                    mu.get_context_hierarchy(node.id)
                                } else {
                                    Vec::new()
                                };

                                builder.add_fact(RetrievedFact {
                                    id: node.id,
                                    content: node.content.clone(),
                                    similarity,
                                    source_mu: mu.id,
                                    source_node: activated.node_id.clone(),
                                    context_hierarchy: context,
                                    depth: node.depth,
                                    timestamp: Some(node.timestamp),
                                });
                            }
                        }
                    }
                }
            }
        }

        builder.set_trees_searched(trees_searched);

        // Build and return result
        let mut result = builder.build();
        
        // Truncate to max_facts and record latency
        result.facts.truncate(self.config.max_facts);
        result.latency_us = start_time.elapsed().as_micros() as u64;
        
        result
    }

    /// Quick semantic search without G-Layer (direct MU access)
    /// 
    /// Useful when you already know which MU to search,
    /// bypassing the contextual reasoning phase.
    pub fn direct_search(
        &self,
        query: &Embedding,
        tree: &crate::mu_tree::MUTree,
        top_k: usize,
    ) -> Vec<RetrievedFact> {
        let search_results = tree.search_semantic(query, top_k);
        
        search_results
            .into_iter()
            .filter(|(_, sim)| *sim >= self.config.fact_threshold)
            .map(|(node, similarity)| {
                let context = if self.config.include_context_hierarchy {
                    tree.get_context_hierarchy(node.id)
                } else {
                    Vec::new()
                };
                
                RetrievedFact {
                    id: node.id,
                    content: node.content.clone(),
                    similarity,
                    source_mu: tree.id,
                    source_node: String::new(),
                    context_hierarchy: context,
                    depth: node.depth,
                    timestamp: Some(node.timestamp),
                }
            })
            .collect()
    }

    /// Get current configuration
    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RetrievalConfig) {
        self.config = config;
    }
}

impl Default for DualRetrieval {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g_layer::{GNode, Triple};
    use crate::mu_tree::MUTree;
    use crate::psi::PsiPointer;
    use ndarray::array;

    #[test]
    fn test_direct_search() {
        let mut tree = MUTree::new("test".into(), 3);
        tree.insert_semantic("The sky is blue".into(), array![0.1, 0.2, 0.9]);
        tree.insert_semantic("Water is wet".into(), array![0.8, 0.1, 0.1]);
        
        let retrieval = DualRetrieval::new();
        let query = array![0.1, 0.2, 0.85];
        
        let results = retrieval.direct_search(&query, &tree, 5);
        
        assert!(!results.is_empty());
        // First result should be most similar to query (the "blue" fact)
        assert!(results[0].content.contains("sky") || results[0].content.contains("blue"));
    }

    #[test]
    fn test_dual_retrieval() {
        // Setup G-Layer
        let mut g_layer = GLayer::new();
        let mut registry = MURegistry::new();
        
        // Create MU-Tree with facts about France
        let mut tree = MUTree::new("facts".into(), 3);
        tree.insert_semantic("Paris is the capital of France".into(), array![0.9, 0.1, 0.0]);
        tree.insert_semantic("France is in Europe".into(), array![0.8, 0.2, 0.0]);
        
        let tree_id = tree.id;
        registry.register(tree);
        
        // Create G-Node with PSI pointer
        let mut node = GNode::new("france".into(), "France".into(), array![0.85, 0.15, 0.0]);
        node.set_psi(PsiPointer::new(tree_id));
        g_layer.add_node(node);
        
        // Query for France-related facts
        let retrieval = DualRetrieval::new();
        let query = array![0.9, 0.1, 0.0];
        
        let result = retrieval.query(&query, &g_layer, &registry);
        
        assert!(result.has_results());
        assert!(result.nodes_activated > 0);
        assert!(result.trees_searched > 0);
    }

    #[test]
    fn test_config_presets() {
        let fast = RetrievalConfig::fast();
        assert!(fast.max_hops < 3);
        assert!(fast.max_facts < 20);
        
        let thorough = RetrievalConfig::thorough();
        assert!(thorough.max_hops > 3);
        assert!(thorough.max_facts > 20);
    }
}
