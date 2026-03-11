//! Similarity functions for embeddings

use super::Embedding;

/// Cosine similarity between two embeddings
///
/// Returns value in range [-1, 1] where 1 is identical direction
#[inline]
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Dot product between two embeddings
#[inline]
pub fn dot_product(a: &Embedding, b: &Embedding) -> f32 {
    a.dot(b)
}

/// Euclidean distance between two embeddings
#[inline]
pub fn euclidean_distance(a: &Embedding, b: &Embedding) -> f32 {
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

/// Squared euclidean distance (faster, no sqrt)
#[inline]
pub fn squared_distance(a: &Embedding, b: &Embedding) -> f32 {
    let diff = a - b;
    diff.dot(&diff)
}

/// Manhattan distance (L1 norm)
#[inline]
pub fn manhattan_distance(a: &Embedding, b: &Embedding) -> f32 {
    (a - b).mapv(f32::abs).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = array![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
        
        let d = array![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }
}
