//! Persistence layer for disk storage

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::g_layer::GLayer;

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base directory for storage
    pub base_path: PathBuf,
    
    /// Whether to auto-save on changes
    pub auto_save: bool,
    
    /// Compression enabled
    pub compress: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("./gt_data"),
            auto_save: false,
            compress: false,
        }
    }
}

/// Persistence errors
#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Path not found: {0}")]
    PathNotFound(String),
}

/// Persistence layer for disk storage
pub struct Persistence {
    config: StorageConfig,
}

impl Persistence {
    /// Create new persistence layer
    pub fn new(config: StorageConfig) -> Result<Self, PersistenceError> {
        // Ensure base directory exists
        fs::create_dir_all(&config.base_path)?;
        
        Ok(Self { config })
    }

    /// Save G-Layer to disk
    pub fn save_g_layer(&self, g_layer: &GLayer) -> Result<(), PersistenceError> {
        let path = self.config.base_path.join("g_layer.json");
        let json = g_layer.to_json()?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load G-Layer from disk
    pub fn load_g_layer(&self) -> Result<GLayer, PersistenceError> {
        let path = self.config.base_path.join("g_layer.json");
        let json = fs::read_to_string(path)?;
        let g_layer = GLayer::from_json(&json)?;
        Ok(g_layer)
    }

    /// Save arbitrary data as JSON
    pub fn save_json<T: Serialize>(&self, name: &str, data: &T) -> Result<(), PersistenceError> {
        let path = self.config.base_path.join(format!("{}.json", name));
        let json = serde_json::to_string_pretty(data)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load arbitrary data from JSON
    pub fn load_json<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Result<T, PersistenceError> {
        let path = self.config.base_path.join(format!("{}.json", name));
        let json = fs::read_to_string(path)?;
        let data = serde_json::from_str(&json)?;
        Ok(data)
    }

    /// Check if data exists
    pub fn exists(&self, name: &str) -> bool {
        let path = self.config.base_path.join(format!("{}.json", name));
        path.exists()
    }

    /// List all saved files
    pub fn list_files(&self) -> Result<Vec<String>, PersistenceError> {
        let entries = fs::read_dir(&self.config.base_path)?;
        
        let files: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        
        Ok(files)
    }

    /// Get storage path
    pub fn base_path(&self) -> &Path {
        &self.config.base_path
    }
}

impl Default for Persistence {
    fn default() -> Self {
        Self::new(StorageConfig::default()).expect("Failed to create default persistence")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let config = StorageConfig {
            base_path: dir.path().to_path_buf(),
            auto_save: false,
            compress: false,
        };
        
        let persistence = Persistence::new(config).unwrap();
        
        // Save some data
        let data = vec!["test1", "test2"];
        persistence.save_json("test_data", &data).unwrap();
        
        // Load it back
        let loaded: Vec<String> = persistence.load_json("test_data").unwrap();
        assert_eq!(loaded, data);
    }
}
