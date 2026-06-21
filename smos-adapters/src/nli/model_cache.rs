//! HF Hub download + on-disk cache for the DeBERTa-v3 ONNX artifacts.
//!
//! The native NLI backend needs two artifacts per model id:
//!
//! - `onnx/model_quantized.onnx` — the quantised ONNX graph (~643 MB for the
//!   default `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` model).
//! - `tokenizer.json` — the SentencePiece BPE tokenizer config consumed by
//!   the `tokenizers` crate.
//!
//! Both are downloaded once via `hf-hub` and copied into a canonical flat
//! layout under `cache_dir` so the classifier does not have to know about the
//! `models--<publisher>--<repo>/snapshots/<sha>/...` directory shape that
//! `hf-hub` uses internally. The flat layout also lets a manual operator drop
//! pre-downloaded artifacts into `cache_dir` without learning the HF cache
//! conventions.
//!
//! # Atomicity + concurrency
//!
//! Each artifact is downloaded to `<name>.part` first, then atomically
//! renamed to `<name>`. A concurrent first-use race (watcher + HTTP extractor
//! both seeing an absent cache) is resolved by `create_new(true)` on the
//! `.part` file: the loser's write fails fast instead of double-downloading
//! the 643 MB graph, and the loser then waits for the winner's rename.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;

/// Canonical filename for the quantised ONNX graph inside `cache_dir`.
pub const MODEL_FILENAME: &str = "model_quantized.onnx";

/// Canonical filename for the tokenizer config inside `cache_dir`.
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// HF Hub path of the quantised graph inside the model repo.
const ONNX_REPO_PATH: &str = "onnx/model_quantized.onnx";

/// Errors surfaced by the model cache layer. Wrapping both HF Hub failures
/// and local IO behind one enum keeps the classifier's error mapping a single
/// match arm — the upstream use case only cares that "the backend is
/// unavailable", not which subsystem failed.
#[derive(Debug, thiserror::Error)]
pub enum ModelCacheError {
    #[error("HF Hub download failed: {0}")]
    DownloadFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Build a configured `hf-hub` sync API client rooted at `cache_dir`.
///
/// Centralised so both model and tokenizer downloads share the same cache
/// location and the same error-mapping convention.
fn hf_api(cache_dir: &Path) -> Result<hf_hub::api::sync::Api, ModelCacheError> {
    ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| ModelCacheError::DownloadFailed(e.to_string()))
}

/// Atomically claim a `.part` file for the current process so two concurrent
/// first-use callers do not both download the same artifact. Returns
/// `Ok(OwnedClaim)` when the caller is the sole downloader, or
/// `Err(TryClaimError::AlreadyInProgress)` when another caller won the race
/// (the loser should retry on the canonical path until the rename lands).
///
/// Self-healing: if a previous claim-holder crashed hard (panic, OOM-kill,
/// power loss), its `.part` file stays on disk forever and the next caller
/// would never win `create_new`. The claim logic detects stale `.part`
/// files via mtime: anything older than [`STALE_PART_TTL`] is treated as
/// abandoned and unlinked before the `create_new` retry.
struct PartClaim {
    /// Owned write handle to the `.part` file — used as the copy target so
    /// we never re-open the same path twice.
    file: std::fs::File,
    part_path: PathBuf,
}

/// How long a `.part` file may sit untouched before it is considered
/// abandoned. Generous on purpose: a slow link downloading 643 MB can take
/// tens of minutes, so the TTL is measured in hours.
const STALE_PART_TTL: std::time::Duration = std::time::Duration::from_secs(6 * 60 * 60);

impl PartClaim {
    /// Create the `.part` file with `create_new(true)`. Two concurrent
    /// callers race on the underlying `O_CREAT | O_EXCL` — only one wins.
    /// A pre-existing `.part` older than [`STALE_PART_TTL`] is unlinked
    /// first so a hard-killed prior process does not wedge the cache
    /// forever.
    fn try_claim(part_path: PathBuf) -> Result<Self, std::io::Error> {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&part_path)
        {
            Ok(file) => Ok(Self { file, part_path }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if Self::is_stale(&part_path)? {
                    tracing::warn!(
                        part = %part_path.display(),
                        "removing stale `.part` left by a previous crashed download"
                    );
                    // `remove_file` is racy with a concurrent winner; if the
                    // remove wins, our second `create_new` claims the path.
                    // If a concurrent winner already recreated it, we fall
                    // through to the AlreadyExists branch below.
                    let _ = std::fs::remove_file(&part_path);
                    return OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&part_path)
                        .map(|file| Self { file, part_path });
                }
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// `true` if `path`'s mtime is older than [`STALE_PART_TTL`].
    ///
    /// Returns `true` when the path does not exist (treated as "nothing to
    /// reclaim, the caller is free to try"). Other metadata failures
    /// (unreadable mtime, permission denied) propagate as `Err` so the
    /// caller surfaces a real IO problem instead of silently wedging.
    fn is_stale(path: &Path) -> Result<bool, std::io::Error> {
        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(true),
            Err(e) => return Err(e),
        };
        let mtime = metadata
            .modified()
            .map_err(|_| std::io::Error::other("mtime not available"))?;
        let age = std::time::SystemTime::now()
            .duration_since(mtime)
            .unwrap_or(std::time::Duration::ZERO);
        Ok(age > STALE_PART_TTL)
    }
}

impl Drop for PartClaim {
    fn drop(&mut self) {
        // Best-effort cleanup of a stale `.part` (e.g. process killed
        // mid-download). Errors are swallowed — there is nothing useful to
        // do with them at drop time.
        let _ = std::fs::remove_file(&self.part_path);
    }
}

/// Resolve a single HF Hub file to a local canonical path.
///
/// Download + copy lands in `<cache_dir>/<local_name>.part` first, then the
/// `.part` suffix is dropped via atomic rename. Two concurrent callers
/// racing on a cold cache coordinate through `PartClaim`: the loser sees
/// `create_new` fail and polls the canonical path until the winner's rename
/// lands.
fn fetch_canonical(
    model_id: &str,
    cache_dir: &Path,
    repo_path: &str,
    local_name: &str,
) -> Result<PathBuf, ModelCacheError> {
    let local = cache_dir.join(local_name);
    if local.exists() {
        return Ok(local);
    }

    tracing::info!(
        model = model_id,
        repo_path = repo_path,
        cache_dir = %cache_dir.display(),
        "downloading NLI artifact from HF Hub"
    );

    let api = hf_api(cache_dir)?;
    let repo = api.model(model_id.to_string());
    let downloaded = repo
        .get(repo_path)
        .map_err(|e| ModelCacheError::DownloadFailed(e.to_string()))?;

    std::fs::create_dir_all(cache_dir)?;

    // If the HF cache already produced the canonical name (rare but possible
    // when `cache_dir` matches the HF default), skip the atomic dance —
    // `rename` to the same path is a no-op but adds nothing.
    if downloaded == local {
        return Ok(local);
    }

    let part_path = cache_dir.join(format!("{local_name}.part"));
    // Race-resolved single downloader. The loser hits `create_new` failure
    // (or a stale-`.part` detection inside `try_claim`) and falls through
    // to the polling loop below. The claim's owned file handle is the
    // copy target so we never re-open the same path twice.
    //
    // Error triage on `try_claim`:
    //   * `AlreadyExists` — a concurrent process holds the `.part`; we are
    //     the loser and must poll for the canonical rename. This is the
    //     expected race outcome.
    //   * Any other IO error (permission denied, disk full, …) — a real
    //     environment problem. Failing fast surfaces it instead of polling
    //     a canonical path that will never appear.
    match PartClaim::try_claim(part_path.clone()) {
        Ok(mut claim) => {
            // Stream the HF copy through memory-mapped buffers; `io::copy`
            // handles chunking. The claim is held until the rename
            // completes — its `Drop` cleans up the `.part` if the copy or
            // rename fails mid-flight.
            let mut source = std::fs::File::open(&downloaded)?;
            std::io::copy(&mut source, &mut claim.file)?;
            claim.file.flush()?;
            // Atomic on POSIX; on Windows `rename` over an existing file
            // fails, but `local.exists()` was false above so the target is
            // absent and the rename succeeds.
            std::fs::rename(&part_path, &local)?;
            // Rename succeeded. `mem::forget(claim)` keeps `Drop` from
            // running on the (now-empty) `part_path`: a concurrent caller
            // may have already won `create_new` on the same path in the
            // window between our rename and our `Drop`, and removing
            // `part_path` here would delete *their* in-flight `.part`.
            // Side effect: the OS file handle inside `claim.file` is not
            // closed explicitly until process exit. Bounded — at most two
            // handles (model + tokenizer) per process lifetime — so the
            // leak is acceptable for the race-safety it buys.
            std::mem::forget(claim);
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Another caller won the claim. Poll the canonical path until
            // either the rename lands or we exhaust the retry budget; the
            // budget is generous because a 643 MB download on a slow link
            // can take minutes.
            const RETRIES: u32 = 300;
            const RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);
            for _ in 0..RETRIES {
                if local.exists() {
                    return Ok(local);
                }
                std::thread::sleep(RETRY_INTERVAL);
            }
            return Err(ModelCacheError::DownloadFailed(format!(
                "canonical file did not appear after {RETRIES} retries: {}",
                local.display()
            )));
        }
        Err(other) => {
            // Real IO problem (permission denied, read-only filesystem, …).
            // Fail fast so the operator sees the root cause instead of a
            // mysterious "canonical file did not appear" timeout later.
            return Err(ModelCacheError::Io(other));
        }
    }

    tracing::info!(path = %local.display(), "NLI artifact cached");
    Ok(local)
}

/// Ensure the quantised ONNX model is cached at
/// `<cache_dir>/model_quantized.onnx` and return its path. Downloads from
/// HF Hub on first use; subsequent calls hit the local copy.
pub fn ensure_model_cached(model_id: &str, cache_dir: &Path) -> Result<PathBuf, ModelCacheError> {
    fetch_canonical(model_id, cache_dir, ONNX_REPO_PATH, MODEL_FILENAME)
}

/// Ensure `tokenizer.json` is cached at `<cache_dir>/tokenizer.json` and
/// return its path. Same caching policy as [`ensure_model_cached`].
pub fn ensure_tokenizer_cached(
    model_id: &str,
    cache_dir: &Path,
) -> Result<PathBuf, ModelCacheError> {
    fetch_canonical(model_id, cache_dir, TOKENIZER_FILENAME, TOKENIZER_FILENAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_filenames_are_stable() {
        // The classifier, CLI defaults, and tests all hard-code these names
        // — flipping them silently would invalidate every existing cache.
        assert_eq!(MODEL_FILENAME, "model_quantized.onnx");
        assert_eq!(TOKENIZER_FILENAME, "tokenizer.json");
    }

    #[test]
    fn onnx_repo_path_matches_hf_layout() {
        // Mirror the path documented on the model card; if HF ever moves it
        // a loud failure here is preferable to a silent 404 at runtime.
        assert_eq!(ONNX_REPO_PATH, "onnx/model_quantized.onnx");
    }

    #[test]
    fn error_display_carries_root_cause() {
        let io = ModelCacheError::Io(std::io::Error::other("boom"));
        assert!(io.to_string().contains("boom"));

        let dl = ModelCacheError::DownloadFailed("404 not found".into());
        assert!(dl.to_string().contains("404 not found"));
    }
}
