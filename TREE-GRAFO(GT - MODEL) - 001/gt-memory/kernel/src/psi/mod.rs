//! # Ψ Protocol Module
//!
//! Symbolic resolution protocol linking G-Layer to MU-Tree Layer.
//! Implements the Ψ pointer and Ψ-Share mechanisms for structural sharing.

mod pointer;
mod share;

pub use pointer::PsiPointer;
pub use share::{PsiShare, MURegistry, PsiShareConfig, ShareStats, ShareReference};

