//! PsiPointer - Symbolic link between G-Layer and MU-Tree

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Ψ Pointer - Symbolic resolution link
///
/// Maps a G-Layer element (node or edge) to a Memory Unit address.
/// Supports offset for sub-tree access (Ψ_offset).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PsiPointer {
    /// Address of the Memory Unit (MU-Tree ID)
    pub mu_address: Uuid,
    
    /// Optional offset path within the MU-Tree
    /// Format: "LRL" = left -> right -> left from root
    pub offset: Option<String>,
    
    /// Whether this is a shared pointer (via Ψ-Share)
    pub is_shared: bool,
    
    /// Reference to original pointer if shared
    pub shared_from: Option<Uuid>,
}

impl PsiPointer {
    /// Create a new Ψ pointer to MU root
    pub fn new(mu_address: Uuid) -> Self {
        Self {
            mu_address,
            offset: None,
            is_shared: false,
            shared_from: None,
        }
    }

    /// Create Ψ pointer with offset to sub-tree
    pub fn with_offset(mu_address: Uuid, offset: String) -> Self {
        Self {
            mu_address,
            offset: Some(offset),
            is_shared: false,
            shared_from: None,
        }
    }

    /// Create a shared pointer (Ψ-Share)
    ///
    /// Used when multiple G-Layer concepts need to reference
    /// the same factual sub-structure
    pub fn create_shared(&self, new_offset: Option<String>) -> Self {
        Self {
            mu_address: self.mu_address,
            offset: new_offset.or_else(|| self.offset.clone()),
            is_shared: true,
            shared_from: Some(self.mu_address),
        }
    }

    /// Create shared pointer to a sub-tree
    pub fn share_subtree(&self, subtree_offset: &str) -> Self {
        let combined_offset = match &self.offset {
            Some(existing) => format!("{}{}", existing, subtree_offset),
            None => subtree_offset.to_string(),
        };
        
        Self {
            mu_address: self.mu_address,
            offset: Some(combined_offset),
            is_shared: true,
            shared_from: Some(self.mu_address),
        }
    }

    /// Get the full path (for resolution)
    pub fn get_full_path(&self) -> (Uuid, Option<&str>) {
        (self.mu_address, self.offset.as_deref())
    }

    /// Check if pointing to root or sub-tree
    pub fn is_root_pointer(&self) -> bool {
        self.offset.is_none()
    }

    /// Get offset depth
    pub fn offset_depth(&self) -> usize {
        self.offset.as_ref().map(|o| o.len()).unwrap_or(0)
    }
}

impl std::fmt::Display for PsiPointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.offset {
            Some(offset) => write!(f, "Ψ[{}:{}]", self.mu_address, offset),
            None => write!(f, "Ψ[{}]", self.mu_address),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_pointer() {
        let id = Uuid::new_v4();
        let ptr = PsiPointer::new(id);
        
        assert_eq!(ptr.mu_address, id);
        assert!(ptr.is_root_pointer());
        assert!(!ptr.is_shared);
    }

    #[test]
    fn test_shared_pointer() {
        let id = Uuid::new_v4();
        let original = PsiPointer::new(id);
        let shared = original.create_shared(Some("LR".to_string()));
        
        assert!(shared.is_shared);
        assert_eq!(shared.shared_from, Some(id));
        assert_eq!(shared.offset, Some("LR".to_string()));
    }

    #[test]
    fn test_share_subtree() {
        let id = Uuid::new_v4();
        let original = PsiPointer::with_offset(id, "L".to_string());
        let shared = original.share_subtree("RR");
        
        assert_eq!(shared.offset, Some("LRR".to_string()));
    }
}
