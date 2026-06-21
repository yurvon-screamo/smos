//! Storage adapters — `SurrealStore` (persistence), `SystemClock` (time),
//! and `SystemIdGenerator` (fresh session ids).

pub mod surreal_schema;
pub mod surreal_store;
pub mod system_clock;
pub mod system_id_generator;
