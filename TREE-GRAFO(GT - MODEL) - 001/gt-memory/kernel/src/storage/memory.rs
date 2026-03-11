//! Memory Tiering for Scalability
//!
//! Implements Section 5.2: "Tiering de Memória para Escalabilidade"
//!
//! From the paper:
//! - "Nível Rápido (Memória RAM/Cache): O G-Layer principal e os nós internos
//!    de alta abstração das MUs devem residir aqui"
//! - "Nível Lento (Armazenamento Persistente): Os nós folha das MU-Trees"
//!
//! This strategy "mitiga gargalos de armazenamento e I/O, assegurando que a fase
//! de raciocínio (G-Layer) seja mantida em alta velocidade"

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use std::time::{Duration, Instant};

use crate::g_layer::GLayer;
use crate::psi::MURegistry;

/// Memory Tier Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierLevel {
    /// Hot tier: RAM/Cache - frequently accessed
    /// Contains: G-Layer, MU roots, recent embeddings
    Hot,
    
    /// Warm tier: SSD/Fast storage - occasionally accessed
    /// Contains: Less-used MU nodes, older embeddings
    Warm,
    
    /// Cold tier: HDD/Archive - rarely accessed  
    /// Contains: Historical data, leaf nodes on demand
    Cold,
}

/// Configuration for memory tiering
#[derive(Debug, Clone)]
pub struct TieringConfig {
    /// Maximum number of MUs to keep in hot tier
    pub max_hot_mus: usize,
    
    /// Time before an MU is demoted to warm tier
    pub hot_to_warm_threshold: Duration,
    
    /// Time before an MU is demoted to cold tier
    pub warm_to_cold_threshold: Duration,
    
    /// Minimum access count to stay in hot tier
    pub min_hot_access_count: usize,
}

impl Default for TieringConfig {
    fn default() -> Self {
        Self {
            max_hot_mus: 100,
            hot_to_warm_threshold: Duration::from_secs(3600), // 1 hour
            warm_to_cold_threshold: Duration::from_secs(86400), // 24 hours
            min_hot_access_count: 5,
        }
    }
}

/// Access tracking for LRU/LFU decisions
#[derive(Debug, Clone)]
struct AccessStats {
    last_access: Instant,
    access_count: usize,
    tier: TierLevel,
}

impl Default for AccessStats {
    fn default() -> Self {
        Self {
            last_access: Instant::now(),
            access_count: 0,
            tier: TierLevel::Hot,
        }
    }
}

/// In-memory storage tier with tiering support
///
/// Implements the von Neumann-aligned memory hierarchy from Section 5.2:
/// - G-Layer always in hot tier (frequently accessed relational skeleton)
/// - MU roots and high-abstraction nodes in hot tier
/// - Leaf nodes can be demoted to slower tiers
pub struct MemoryTier {
    /// The G-Layer (always in hot tier)
    /// From paper: "O G-Layer principal (o esqueleto relacional, que é
    /// frequentemente acessado) ... devem residir aqui"
    pub g_layer: Arc<RwLock<GLayer>>,
    
    /// MU Registry (roots and embeddings in hot tier)
    pub registry: Arc<RwLock<MURegistry>>,
    
    /// Access statistics for tiering decisions
    access_stats: HashMap<Uuid, AccessStats>,
    
    /// Tiering configuration
    config: TieringConfig,
    
    /// MUs currently in hot tier
    hot_tier: Vec<Uuid>,
    
    /// MUs currently in warm tier
    warm_tier: Vec<Uuid>,
    
    /// MUs currently in cold tier (would be on disk in production)
    cold_tier: Vec<Uuid>,
}

impl MemoryTier {
    /// Create new memory tier with default config
    pub fn new() -> Self {
        Self::with_config(TieringConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: TieringConfig) -> Self {
        Self {
            g_layer: Arc::new(RwLock::new(GLayer::new())),
            registry: Arc::new(RwLock::new(MURegistry::new())),
            access_stats: HashMap::new(),
            config,
            hot_tier: Vec::new(),
            warm_tier: Vec::new(),
            cold_tier: Vec::new(),
        }
    }

    /// Get G-Layer reference (always hot)
    pub fn g_layer(&self) -> Arc<RwLock<GLayer>> {
        self.g_layer.clone()
    }

    /// Get registry reference
    pub fn registry(&self) -> Arc<RwLock<MURegistry>> {
        self.registry.clone()
    }

    /// Record access to an MU and update tier
    pub fn record_access(&mut self, mu_id: Uuid) {
        let entry = self.access_stats.entry(mu_id).or_default();
        entry.last_access = Instant::now();
        entry.access_count += 1;
        
        // Promote to hot tier if frequently accessed
        if entry.access_count >= self.config.min_hot_access_count {
            if entry.tier != TierLevel::Hot {
                self.promote_to_hot(mu_id);
            }
        }
    }

    /// Promote an MU to hot tier
    fn promote_to_hot(&mut self, mu_id: Uuid) {
        // Remove from other tiers
        self.warm_tier.retain(|id| *id != mu_id);
        self.cold_tier.retain(|id| *id != mu_id);
        
        // Add to hot if not already there
        if !self.hot_tier.contains(&mu_id) {
            self.hot_tier.push(mu_id);
        }
        
        // Update stats
        if let Some(stats) = self.access_stats.get_mut(&mu_id) {
            stats.tier = TierLevel::Hot;
        }
        
        // Evict if over capacity
        self.enforce_hot_tier_limit();
    }

    /// Enforce hot tier size limit using LRU
    fn enforce_hot_tier_limit(&mut self) {
        while self.hot_tier.len() > self.config.max_hot_mus {
            // Find least recently used
            let lru = self.hot_tier
                .iter()
                .min_by_key(|id| {
                    self.access_stats.get(id)
                        .map(|s| s.last_access)
                        .unwrap_or(Instant::now())
                })
                .cloned();
            
            if let Some(mu_id) = lru {
                self.demote_to_warm(mu_id);
            }
        }
    }

    /// Demote an MU to warm tier
    fn demote_to_warm(&mut self, mu_id: Uuid) {
        self.hot_tier.retain(|id| *id != mu_id);
        
        if !self.warm_tier.contains(&mu_id) {
            self.warm_tier.push(mu_id);
        }
        
        if let Some(stats) = self.access_stats.get_mut(&mu_id) {
            stats.tier = TierLevel::Warm;
        }
    }

    /// Demote an MU to cold tier
    fn demote_to_cold(&mut self, mu_id: Uuid) {
        self.warm_tier.retain(|id| *id != mu_id);
        
        if !self.cold_tier.contains(&mu_id) {
            self.cold_tier.push(mu_id);
        }
        
        if let Some(stats) = self.access_stats.get_mut(&mu_id) {
            stats.tier = TierLevel::Cold;
        }
    }

    /// Run tier maintenance (should be called periodically)
    pub fn maintain_tiers(&mut self) {
        let now = Instant::now();
        
        // Find MUs to demote from hot to warm
        let hot_to_demote: Vec<Uuid> = self.hot_tier
            .iter()
            .filter(|id| {
                self.access_stats.get(id)
                    .map(|s| now.duration_since(s.last_access) > self.config.hot_to_warm_threshold)
                    .unwrap_or(true)
            })
            .cloned()
            .collect();
        
        for id in hot_to_demote {
            self.demote_to_warm(id);
        }
        
        // Find MUs to demote from warm to cold
        let warm_to_demote: Vec<Uuid> = self.warm_tier
            .iter()
            .filter(|id| {
                self.access_stats.get(id)
                    .map(|s| now.duration_since(s.last_access) > self.config.warm_to_cold_threshold)
                    .unwrap_or(true)
            })
            .cloned()
            .collect();
        
        for id in warm_to_demote {
            self.demote_to_cold(id);
        }
    }

    /// Get tier for an MU
    pub fn get_tier(&self, mu_id: &Uuid) -> TierLevel {
        self.access_stats.get(mu_id)
            .map(|s| s.tier)
            .unwrap_or(TierLevel::Cold)
    }

    /// Get access count for an MU
    pub fn get_access_count(&self, mu_id: &Uuid) -> usize {
        self.access_stats.get(mu_id)
            .map(|s| s.access_count)
            .unwrap_or(0)
    }

    /// Reset access tracking
    pub fn reset_access_tracking(&mut self) {
        self.access_stats.clear();
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        let g_layer = self.g_layer.read();
        let registry = self.registry.read();
        
        MemoryStats {
            g_layer_nodes: g_layer.node_count(),
            g_layer_edges: g_layer.edge_count(),
            registered_mus: registry.len(),
            hot_tier_mus: self.hot_tier.len(),
            warm_tier_mus: self.warm_tier.len(),
            cold_tier_mus: self.cold_tier.len(),
            max_hot_mus: self.config.max_hot_mus,
        }
    }
}

impl Default for MemoryTier {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub g_layer_nodes: usize,
    pub g_layer_edges: usize,
    pub registered_mus: usize,
    pub hot_tier_mus: usize,
    pub warm_tier_mus: usize,
    pub cold_tier_mus: usize,
    pub max_hot_mus: usize,
}

impl MemoryStats {
    /// Calculate hot tier utilization percentage
    pub fn hot_tier_utilization(&self) -> f32 {
        if self.max_hot_mus == 0 {
            return 0.0;
        }
        (self.hot_tier_mus as f32 / self.max_hot_mus as f32) * 100.0
    }

    /// Calculate total MUs across all tiers
    pub fn total_tiered_mus(&self) -> usize {
        self.hot_tier_mus + self.warm_tier_mus + self.cold_tier_mus
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tier_creation() {
        let tier = MemoryTier::new();
        let stats = tier.stats();
        
        assert_eq!(stats.g_layer_nodes, 0);
        assert_eq!(stats.hot_tier_mus, 0);
    }

    #[test]
    fn test_access_tracking() {
        let mut tier = MemoryTier::new();
        let mu_id = Uuid::new_v4();
        
        for _ in 0..10 {
            tier.record_access(mu_id);
        }
        
        assert_eq!(tier.get_access_count(&mu_id), 10);
    }

    #[test]
    fn test_tier_promotion() {
        let mut tier = MemoryTier::with_config(TieringConfig {
            min_hot_access_count: 3,
            ..Default::default()
        });
        
        let mu_id = Uuid::new_v4();
        
        // Initially not in hot tier
        assert_eq!(tier.get_tier(&mu_id), TierLevel::Cold);
        
        // Access multiple times
        for _ in 0..5 {
            tier.record_access(mu_id);
        }
        
        // Should now be promoted to hot
        assert_eq!(tier.get_tier(&mu_id), TierLevel::Hot);
    }
}
