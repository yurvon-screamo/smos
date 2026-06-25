//! Storage adapters — `SurrealStore` (persistence), `SystemClock` (time),
//! and `SystemIdGenerator` (fresh session ids).
//!
//! `SurrealStore` is a facade (R6): the constructor + connection + migration
//! helpers live in [`surreal_store`], the row↔domain conversion helpers in
//! `mapping` and `rows`, the vector-search passes (HNSW + brute-force) in
//! `vector_search`, and the port-trait implementations in `fact_repository`
//! and `session_repository`. The split is internal;
//! `SurrealStore`'s public API surface (and the trait impl signatures) is
//! identical to the pre-split module.

mod fact_repository;
mod mapping;
mod rows;
mod session_repository;
mod vector_search;

pub mod surreal_schema;
pub mod surreal_store;
pub mod system_clock;
pub mod system_id_generator;
