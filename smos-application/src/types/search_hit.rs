//! Vector-search hit DTO.
//!
//! Mirrors the POC `SearchHit` (`smos/storage.py:36`): id, document, metadata,
//! distance. Metadata is broken out into a typed sub-struct so downstream
//! retrieval-planning logic can read confidence / heat / validity without
//! string-keyed lookups, but the field stays round-trippable as JSON for
//! adapter convenience.

use serde::{Deserialize, Serialize};
use smos_domain::{FactId, MemoryKey};

/// One row returned by `FactRepository::search_similar`.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub id: FactId,
    pub document: String,
    pub memory_key: MemoryKey,
    pub metadata: SearchHitMetadata,
}

/// Strongly-typed view over the POC's `dict[str, object]` metadata bag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHitMetadata {
    /// `accepted` / `pending` / `rejected`.
    pub status: String,
    /// Stored confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// ISO-8601 string of the validity tombstone, or `None` if the fact is
    /// still current. Stored as a string so the row stays self-describing
    /// across adapters without binding to a specific datetime crate.
    pub valid_until: Option<String>,
    /// Heat base value `[0.0, 1.0]`; §7 decay uses this as the seed.
    pub heat_base: f32,
    /// Last-access unix timestamp in seconds. The field is typed as `f32` for
    /// wire compatibility with downstream JSON consumers that emit fractional
    /// seconds, but the SurrealStore adapter currently truncates to whole
    /// seconds (`surreal_store::SearchSimilarRow::to_hit` parses an ISO
    /// datetime and stores `ts.as_unix_secs() as f32`). Treat the value as
    /// second-precision until the storage layer gains sub-second support.
    pub last_access_at: f32,
    /// Cosine distance reported by the vector store. Lower = more similar.
    /// `None` when the underlying store did not surface a distance.
    pub distance: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> SearchHitMetadata {
        SearchHitMetadata {
            status: "accepted".into(),
            confidence: 0.85,
            valid_until: None,
            heat_base: 1.0,
            last_access_at: 1_700_000_000.0,
            distance: Some(0.12),
        }
    }

    #[test]
    fn metadata_roundtrips_through_serde() {
        let meta = sample_metadata();
        let json = serde_json::to_string(&meta).unwrap();
        let back: SearchHitMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, back);
    }

    #[test]
    fn metadata_serialises_optional_valid_until_as_null_when_absent() {
        let meta = sample_metadata();
        let v: serde_json::Value = serde_json::to_value(&meta).unwrap();
        assert_eq!(v["valid_until"], serde_json::Value::Null);
    }

    #[test]
    fn metadata_serialises_optional_distance_as_number_when_present() {
        let meta = sample_metadata();
        let v: serde_json::Value = serde_json::to_value(&meta).unwrap();
        // f32 → f64 widening can introduce tiny representation drift, so
        // compare with tolerance rather than strict equality.
        let got = v["distance"].as_f64().unwrap_or(f64::NAN);
        assert!((got - 0.12).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn metadata_supports_tombstoned_fact() {
        let meta = SearchHitMetadata {
            status: "accepted".into(),
            confidence: 0.9,
            valid_until: Some("2027-01-01T00:00:00Z".into()),
            heat_base: 0.4,
            last_access_at: 1_700_000_050.0,
            distance: None,
        };
        let v: serde_json::Value = serde_json::to_value(&meta).unwrap();
        assert_eq!(v["valid_until"], "2027-01-01T00:00:00Z");
        assert_eq!(v["distance"], serde_json::Value::Null);
    }
}
