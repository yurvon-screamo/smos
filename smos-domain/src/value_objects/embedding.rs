//! `Embedding` — non-empty vector of floats (model-dependent dimensionality).

use crate::error::DomainError;
use crate::value_objects::Cosine;
use serde::{Deserialize, Serialize};

/// Dense embedding of a fact or topic.
///
/// `EXPECTED_DIM` documents the reference dimensionality (Jina v5: 1024d), but
/// construction does *not* enforce it: models change, vector stores adapt, and
/// the domain layer must not crash when it sees a 768d embedding. Empty vectors
/// are rejected because cosine similarity is undefined for them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Reference dimensionality of the Jina v5 small-retrieval model.
    pub const EXPECTED_DIM: usize = 1024;

    pub fn new(v: Vec<f32>) -> Result<Self, DomainError> {
        if v.is_empty() {
            return Err(DomainError::EmptyEmbedding);
        }
        Ok(Self(v))
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Cosine similarity with another embedding.
    ///
    /// Edge cases:
    /// - Empty input or length mismatch → `0.0` (cannot compare).
    /// - Either vector has zero norm → `0.0` (undefined, treated as orthogonal).
    pub fn cosine(&self, other: &Embedding) -> Cosine {
        let a = self.as_slice();
        let b = other.as_slice();
        if a.is_empty() || b.is_empty() || a.len() != b.len() {
            return Cosine::from_raw_clamped(0.0);
        }
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na == 0.0 || nb == 0.0 {
            return Cosine::from_raw_clamped(0.0);
        }
        let denom = na.sqrt() * nb.sqrt();
        Cosine::from_raw_clamped(dot / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DomainError;

    #[test]
    fn new_rejects_empty() {
        assert!(matches!(
            Embedding::new(Vec::new()),
            Err(DomainError::EmptyEmbedding)
        ));
    }

    #[test]
    fn new_accepts_non_empty() {
        let e = Embedding::new(vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(e.dim(), 3);
        assert_eq!(e.as_slice(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn expected_dim_is_1024() {
        assert_eq!(Embedding::EXPECTED_DIM, 1024);
    }

    #[test]
    fn dim_matches_input_length() {
        let v: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let e = Embedding::new(v.clone()).unwrap();
        assert_eq!(e.dim(), v.len());
    }

    #[test]
    fn cosine_identical_unit_vectors_score_one() {
        let v = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        assert!((v.cosine(&v).value() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors_score_zero() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![0.0, 1.0]).unwrap();
        assert!(a.cosine(&b).value().abs() < 1e-5);
    }

    #[test]
    fn cosine_opposite_vectors_score_minus_one() {
        let a = Embedding::new(vec![1.0, 0.0]).unwrap();
        let b = Embedding::new(vec![-1.0, 0.0]).unwrap();
        assert!((a.cosine(&b).value() + 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_mismatched_lengths_yield_zero() {
        let a = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        let b = Embedding::new(vec![1.0, 2.0]).unwrap();
        assert_eq!(a.cosine(&b).value(), 0.0);
    }

    #[test]
    fn cosine_zero_norm_yields_zero() {
        let zero = Embedding::new(vec![0.0, 0.0]).unwrap();
        let other = Embedding::new(vec![1.0, 1.0]).unwrap();
        assert_eq!(zero.cosine(&other).value(), 0.0);
    }

    #[test]
    fn cosine_known_value_for_non_trivial_vectors() {
        // cos([1,0,1], [0,1,1]) = 1 / (sqrt(2) * sqrt(2)) = 0.5
        let a = Embedding::new(vec![1.0, 0.0, 1.0]).unwrap();
        let b = Embedding::new(vec![0.0, 1.0, 1.0]).unwrap();
        assert!((a.cosine(&b).value() - 0.5).abs() < 1e-5);
    }
}
