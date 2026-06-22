//! Git-backed memory sync (placeholder; fleshed out by the submodules).
//!
//! See [`manager::GitSyncManager`] for the high-level API and
//! [`format::FactFrontmatter`] for the on-disk layout.

pub mod export;
pub mod format;
pub mod import;
pub mod manager;

#[cfg(test)]
mod format_tests;
#[cfg(test)]
pub mod test_support;

pub use format::FactFrontmatter;
pub use manager::GitSyncManager;
