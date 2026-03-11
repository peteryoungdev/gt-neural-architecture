//! # Storage Module
//!
//! Memory tiering and persistence for G-T Architecture.

mod memory;
mod persistence;

pub use memory::{MemoryTier, TierLevel, MemoryStats, TieringConfig};
pub use persistence::{Persistence, StorageConfig};

