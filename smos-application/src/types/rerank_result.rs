//! Rerank-result DTO (POC `smos/models.py:133-139`).

use serde::{Deserialize, Serialize};

/// One reranked document with original index preserved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    /// Index of the document in the input `documents: &[String]` slice. Keeps
    /// the reranker stateless about the caller's domain objects.
    pub index: usize,
    /// Cross-encoder relevance score (higher = more relevant).
    pub score: f32,
    /// Echoed document text for convenience.
    pub document: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_serde() {
        let r = RerankResult {
            index: 3,
            score: 0.91,
            document: "Rust is memory-safe".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: RerankResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn preserves_index_for_caller_side_mapping() {
        let r = RerankResult {
            index: 7,
            score: 0.0,
            document: String::new(),
        };
        assert_eq!(r.index, 7);
    }
}
