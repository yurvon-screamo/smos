//! GGUF model download for `smos init`.
//!
//! Downloads the three required GGUF models (extraction LLM, embedding,
//! reranker) from HuggingFace into `~/.smos/models/` using the same `hf-hub`
//! crate that backs the NLI model cache. The canonical local filenames match
//! the `[llama_cpp.*].model_path` defaults shipped in
//! [`super::init_defaults::DEFAULT_CONFIG_TOML`], so a successful init needs
//! no further config edits before `smos serve` can auto-launch every role.
//!
//! # Why sync, not async
//!
//! `hf-hub` with the `ureq` feature (already enabled for the NLI backend)
//! exposes a synchronous download API. `run_init` is async only because some
//! probes (HTTP health, SurrealDB connect) need it; the GGUF download blocks
//! the caller like the filesystem bootstrap does. For a one-shot CLI wizard
//! there is nothing else to interleave with, so blocking is fine and avoids
//! the `spawn_blocking` indirection.
//!
//! # Disk strategy
//!
//! `hf-hub` is pointed at a throwaway `.hf_cache` subdir under `models/`,
//! then the downloaded snapshot is copied to a `<name>.part` sidecar and
//! atomically renamed to the canonical path (so an interrupted copy never
//! leaves a truncated file masquerading as a valid cache), and the cache
//! subdir is removed. Final disk usage is 1× the file size (the HF cache
//! does not persist), at the cost of a full re-download on loss —
//! acceptable for a first-time setup tool where the operator re-runs
//! `smos init` to recover.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;

use crate::paths::SmosPaths;

/// One required GGUF model: HF repo, the file inside it, and the canonical
/// local name it lands under `~/.smos/models/`.
struct GgufModelSpec {
    role: &'static str,
    repo_id: &'static str,
    remote_filename: &'static str,
    local_filename: &'static str,
}

/// The single source of truth for what `smos init` downloads. The
/// `local_filename` values MUST stay in lock-step with the
/// `[llama_cpp.*].model_path` basenames in
/// [`super::init_defaults::DEFAULT_CONFIG_TOML`] and
/// [`crate::llama_server::config::LlamaCppConfig::default`]; the
/// [`local_filenames_match_llama_cpp_defaults`] test guards against drift.
const REQUIRED_GGUF_MODELS: &[GgufModelSpec] = &[
    GgufModelSpec {
        role: "extraction",
        repo_id: "unsloth/Qwen3.5-2B-MTP-GGUF",
        remote_filename: "Qwen3.5-2B-Q5_K_M.gguf",
        local_filename: "qwen3.5-2b-q5_k_m.gguf",
    },
    GgufModelSpec {
        role: "embedding",
        repo_id: "jinaai/jina-embeddings-v5-text-small-retrieval-GGUF",
        remote_filename: "v5-small-retrieval-Q8_0.gguf",
        local_filename: "jina-embeddings-v5.gguf",
    },
    GgufModelSpec {
        role: "reranker",
        repo_id: "DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF",
        remote_filename: "Qwen.Qwen3-Reranker-0.6B.Q8_0.gguf",
        local_filename: "qwen3-reranker.gguf",
    },
];

/// Download every required GGUF into `~/.smos/models/`, printing one
/// ✓ / ✗ row per model. Never aborts the setup wizard on a single failure:
/// a network blip or a private HF repo is reported with a remediation hint
/// so the operator fixes it and re-runs `smos init` (cached files are
/// skipped, so only the failed one is retried).
pub(super) fn download_gguf_models(paths: &SmosPaths) {
    let models_dir = &paths.models;
    let mut any_failed = false;

    for spec in REQUIRED_GGUF_MODELS {
        match download_one(spec, models_dir) {
            Ok(DownloadOutcome::Downloaded) => {
                println!("  ✓ {} ({}) — downloaded", spec.local_filename, spec.role);
            }
            Ok(DownloadOutcome::Cached) => {
                println!(
                    "  ✓ {} ({}) — already present",
                    spec.local_filename, spec.role
                );
            }
            Err(e) => {
                any_failed = true;
                println!("  ✗ {} ({}) — {e}", spec.local_filename, spec.role);
                println!(
                    "    Source: hf.co/{}/{}",
                    spec.repo_id, spec.remote_filename
                );
            }
        }
    }

    if any_failed {
        println!("  ⚠ Some models failed to download. Re-run `smos init` to retry cached-skip,");
        println!(
            "    or fetch them manually from HuggingFace into {}",
            models_dir.display()
        );
    }
}

/// Per-model download outcome. `Cached` short-circuits before any network
/// IO; `Downloaded` means the file was fetched and placed at the canonical
/// path this call.
enum DownloadOutcome {
    Cached,
    Downloaded,
}

/// Download one model if absent. Returns `Cached` when the canonical file
/// already exists (operator re-running init), `Downloaded` on success.
fn download_one(spec: &GgufModelSpec, models_dir: &Path) -> Result<DownloadOutcome, String> {
    let canonical = models_dir.join(spec.local_filename);
    if canonical.exists() {
        return Ok(DownloadOutcome::Cached);
    }

    println!("  ⬇ Downloading {} ({})...", spec.local_filename, spec.role);

    let hf_cache = models_dir.join(".hf_cache");
    std::fs::create_dir_all(&hf_cache).map_err(|e| e.to_string())?;

    let downloaded = fetch_from_hf(spec, &hf_cache)?;
    place_canonical(&downloaded, &canonical)?;

    // Best-effort cleanup of the throwaway HF cache. Errors are swallowed:
    // a leftover cache subdir only wastes disk, it does not affect the
    // canonical file the next step probes.
    let _ = std::fs::remove_dir_all(&hf_cache);

    Ok(DownloadOutcome::Downloaded)
}

/// Resolve the HF snapshot path for one model file. The HF cache is rooted
/// at `hf_cache` so the cleanup in [`download_one`] can wipe it in one
/// `remove_dir_all` after the canonical copy lands.
fn fetch_from_hf(spec: &GgufModelSpec, hf_cache: &Path) -> Result<PathBuf, String> {
    let api = ApiBuilder::new()
        .with_cache_dir(hf_cache.to_path_buf())
        .build()
        .map_err(|e| e.to_string())?;
    let repo = api.model(spec.repo_id.to_string());
    repo.get(spec.remote_filename).map_err(|e| e.to_string())
}

/// Copy the HF snapshot to the canonical path atomically. Writes to a
/// `<canonical>.part` sidecar first, then renames it into place — a
/// rename is atomic on both POSIX and Windows when the destination does
/// not exist (guaranteed by [`download_one`]'s prior `canonical.exists()`
/// check). If the process is interrupted mid-copy, the `.part` is left
/// behind but the canonical path never appears, so the next `smos init`
/// re-downloads instead of trusting a truncated file as a valid cache.
///
/// Always copies (never renames the snapshot directly) because `hf-hub`
/// may expose the snapshot as a symlink to a blob on Unix, and renaming a
/// symlink moves the link rather than the data — leaving the canonical
/// path dangling. A plain `std::fs::copy` reads through the symlink and
/// produces a self-contained file. The peak disk cost (snapshot + `.part`
/// coexisting briefly) is acceptable for a one-time setup download.
fn place_canonical(source: &Path, dest: &Path) -> Result<(), String> {
    let part = part_path(dest);
    // Clear a stale `.part` from a previous interrupted run before writing
    // a fresh one. Absent-file is the common case; a removal failure (e.g.
    // permission denied) surfaces later as a copy error with a clear path.
    let _ = std::fs::remove_file(&part);
    std::fs::copy(source, &part).map_err(|e| format!("write {}: {e}", dest.display()))?;
    std::fs::rename(&part, dest).map_err(|e| format!("finalize {}: {e}", dest.display()))?;
    Ok(())
}

/// Build the `<canonical>.part` sidecar path used by [`place_canonical`].
fn part_path(canonical: &Path) -> PathBuf {
    let mut name = canonical.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_server::LlamaCppConfig;

    /// Acquire the workspace-wide env-test lock —
    /// `LlamaCppConfig::default()` resolves model paths through
    /// `SmosPaths::resolve()`, which reads `SMOS_HOME`.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    /// `REQUIRED_GGUF_MODELS` is the single source of truth for what `smos
    /// init` downloads, and `LlamaCppConfig::default()` is the single source
    /// of truth for what `smos serve` auto-launches. A drift between the two
    /// would make init download a file the launcher never reads (or vice
    /// versa), so pin every `local_filename` against the matching role's
    /// `model_path` basename.
    #[test]
    fn local_filenames_match_llama_cpp_defaults() {
        let _g = lock();
        let cfg = LlamaCppConfig::default();

        for spec in REQUIRED_GGUF_MODELS {
            let configured = match spec.role {
                "extraction" => &cfg.extraction.model_path,
                "embedding" => &cfg.embedding.model_path,
                "reranker" => &cfg.reranker.model_path,
                other => panic!("unknown role {other:?} in REQUIRED_GGUF_MODELS"),
            };
            let basename = Path::new(configured)
                .file_name()
                .expect("model_path has a basename")
                .to_string_lossy();
            assert_eq!(
                basename, spec.local_filename,
                "role `{}`: init downloads `{}` but config points at `{}`",
                spec.role, spec.local_filename, basename
            );
        }
    }

    /// Each role appears exactly once. A duplicate would make one download
    /// silently mask another (or overwrite a sibling role's file).
    #[test]
    fn roles_are_unique() {
        let mut roles: Vec<&str> = REQUIRED_GGUF_MODELS.iter().map(|s| s.role).collect();
        roles.sort_unstable();
        let before = roles.len();
        roles.dedup();
        assert_eq!(
            before,
            roles.len(),
            "duplicate role in REQUIRED_GGUF_MODELS"
        );
    }

    /// `part_path` appends `.part` to the canonical name. Renaming the
    /// snapshot directly (instead of copying through this sidecar) would
    /// break on Unix where `hf-hub` exposes snapshots as symlinks to blobs,
    /// so the sidecar is load-bearing — pin its shape so a refactor cannot
    /// silently drop the atomic-write guarantee.
    #[test]
    fn part_path_appends_part_suffix() {
        let p = part_path(std::path::Path::new("/m/qwen3.5-2b-q5_k_m.gguf"));
        assert_eq!(
            p.file_name().unwrap(),
            "qwen3.5-2b-q5_k_m.gguf.part",
            "part_path must append .part without touching the original extension"
        );
    }
}
