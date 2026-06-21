//! Domain enums shared across the SMOS pipeline.

pub mod fact_status;
pub mod fact_type;
pub mod merge_reason;
pub mod nli_label;

pub use fact_status::FactStatus;
pub use fact_type::FactType;
pub use merge_reason::MergeReason;
pub use nli_label::NliLabel;
