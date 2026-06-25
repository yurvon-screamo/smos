//! Runtime download + on-disk cache for the ONNX Runtime shared library.
//!
//! `ort` is built with `load-dynamic`, so the binary does NOT link an
//! ONNX Runtime shared library at build time. Instead, the matching DLL
//! for the detected [`DeviceKind`] is fetched on first use and cached
//! under `cache_dir/<device>/<dll_name>`. The path is then handed to
//! [`crate::nli::device::build_session`], which calls `ort::init_from`.
//!
//! # Artifact sources
//!
//! | Device    | Platform        | Source                                                |
//! |-----------|-----------------|-------------------------------------------------------|
//! | `Cpu`     | win-x64         | GitHub release `.zip`                                 |
//! | `Cpu`     | linux-x64       | GitHub release `.tgz`                                 |
//! | `Cpu`     | osx-arm64       | GitHub release `.tgz` (CoreML EP bundled)             |
//! | `Cuda`    | win-x64/linux   | GitHub release `-gpu` archive                         |
//! | `DirectML`| win-x64         | NuGet v3 flat-container `.nupkg` (DML is no longer a  |
//! |           |                 | github release asset starting with ORT 1.20.x)        |
//! | `Metal`   | osx-arm64       | Same `.tgz` as `Cpu` (CoreML EP is bundled in CPU)    |
//!
//! CPU is intentionally NOT compiled in — every device downloads its DLL
//! because `load-dynamic` never links. The `Metal` variant reuses the
//! CPU ort build (CoreML ships inside it); the subdir is named `metal`
//! for operator clarity even though the bytes match `cpu/`.
//!
//! # Integrity
//!
//! Downloads go over HTTPS with a 5-minute timeout and a 1 GiB response
//! cap. Extracted files are written into a per-extraction temp directory
//! and renamed into place atomically once the canonical DLL is verified
//! present, so a partial download or extraction crash never produces a
//! half-populated cache that passes the next-startup `exists()` check.
//!
//! Subresource-integrity verification (SHA-256 digest pinned per release
//! and compared before extraction) is the documented follow-up: GitHub
//! publishes per-asset digests via its releases API and would close the
//! CDN-tamper / corporate-MITM vector that bare HTTPS leaves open. Not
//! implemented in this revision to keep the module inside review size.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::device::DeviceKind;

/// ONNX Runtime version downloaded by this module.
///
/// Pinned rather than floating because the ort-rs bindings (=2.0.0-rc.12)
/// target a specific C-API surface; a silent ORT upgrade could expose
/// symbol mismatches that only surface at session-build time on a user
/// machine. Bump explicitly after verifying compatibility.
///
/// v1.24.0 was skipped upstream — the first published release of the
/// 1.24 line is v1.24.1 (Feb 2026).
///
/// When bumping, re-verify every artifact name below against the
/// release page: Microsoft renamed assets mid-1.20.x (see
/// onnxruntime#22925), moved the DirectML drop to NuGet-only by 1.20.x,
/// and starting with v1.26 renamed CUDA packages to
/// `onnxruntime-{platform}-gpu_cudaNN-{version}.{ext}` (the bare
/// `-gpu-` form used below is scheduled for removal once CUDA 12 is
/// dropped, see the v1.27.0 release notes).
const ORT_VERSION: &str = "1.24.1";

/// Hard cap on a single archive download. The largest artifact in the
/// 1.24.x line is the Windows CUDA bundle at ~280 MB; 1 GiB is a
/// generous ceiling that still rejects an obvious malicious/redirected
/// response trying to exhaust memory.
const MAX_DOWNLOAD_BYTES: usize = 1024 * 1024 * 1024;

/// Per-request timeout. Generous because the CUDA archive on a slow
/// link can take minutes, but bounded so a hung CDN connection surfaces
/// as an error instead of an indefinite startup stall.
const DOWNLOAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Archive container the upstream artifact ships in. Drives the extractor
/// dispatch in [`extract_archive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArchiveFormat {
    Zip,
    Tgz,
}

impl ArchiveFormat {
    fn extension(self) -> &'static str {
        match self {
            Self::Zip => "zip",
            Self::Tgz => "tgz",
        }
    }
}

/// Resolved download description for one (device, platform) pair.
struct OrtArtifact {
    url: String,
    /// Canonical leaf name ort expects (e.g. `onnxruntime.dll`,
    /// `libonnxruntime.so`, `libonnxruntime.dylib`).
    dll_name: &'static str,
    format: ArchiveFormat,
    /// Subdirectory under the cache root. Per-device so a host that
    /// flips `[nli_backend].device` does not invalidate another
    /// device's cached DLL.
    cache_subdir: &'static str,
}

/// Map the current `target_os`/`target_arch` to the platform token used
/// in Microsoft's release artifact names plus the canonical DLL leaf
/// name and archive container.
///
/// Apple Intel Macs are intentionally rejected: upstream ORT dropped
/// `osx-x86_64` artifacts starting with the 1.24.x line ("x86_64
/// binaries for macOS/iOS are no longer provided and minimum macOS is
/// raised to 14.0", see the v1.24.1 release notes). An Intel Mac user
/// gets a clear startup error rather than a mysterious 404 mid-download.
fn current_platform() -> Result<(&'static str, &'static str, ArchiveFormat)> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Ok(("win-x64", "onnxruntime.dll", ArchiveFormat::Zip)),
        ("linux", "x86_64") => Ok(("linux-x64", "libonnxruntime.so", ArchiveFormat::Tgz)),
        ("macos", "aarch64") => Ok(("osx-arm64", "libonnxruntime.dylib", ArchiveFormat::Tgz)),
        ("macos", "x86_64") => anyhow::bail!(
            "ONNX Runtime 1.24+ no longer publishes osx-x86_64 binaries; \
             Apple Intel Macs are unsupported. Upgrade to Apple Silicon or \
             pin ORT_VERSION to 1.23.x."
        ),
        (os, arch) => anyhow::bail!("unsupported ORT platform: {os}-{arch}"),
    }
}

/// Resolve the download URL, archive format, and cache subdir for `device`
/// on the current platform.
///
/// DirectML is fetched from the NuGet v3 flat-container API because
/// Microsoft stopped publishing DML as a github release zip starting
/// with the 1.20.x line — the `.nupkg` is itself a zip and is extracted
/// with the same logic as the Windows CPU artifact (see
/// [`extract_archive`]). v3 flat-container is preferred over the legacy
/// v2 endpoint because v2 is in deprecation.
fn artifact_for(device: DeviceKind) -> Result<OrtArtifact> {
    let (platform, dll_name, format) = current_platform()?;
    let ext = format.extension();

    let (cache_subdir, url) = match device {
        DeviceKind::DirectML if platform == "win-x64" => (
            "directml",
            // NuGet v3 flat-container requires lowercase id + version.
            format!(
                "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/{ORT_VERSION}/microsoft.ml.onnxruntime.directml.{ORT_VERSION}.nupkg"
            ),
        ),
        DeviceKind::Cuda if matches!(platform, "win-x64" | "linux-x64") => (
            "cuda",
            format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-{platform}-gpu-{ORT_VERSION}.{ext}"
            ),
        ),
        DeviceKind::Metal if platform == "osx-arm64" => (
            // CoreML EP is bundled in the CPU ort build for macOS; no
            // separate Metal package exists. The subdir is named
            // `metal` for operator clarity even though the bytes match
            // `cpu/`.
            "metal",
            format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-{platform}-{ORT_VERSION}.{ext}"
            ),
        ),
        DeviceKind::Cpu => (
            "cpu",
            format!(
                "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-{platform}-{ORT_VERSION}.{ext}"
            ),
        ),
        _ => anyhow::bail!("no ORT artifact for device {device:?} on platform {platform}"),
    };

    Ok(OrtArtifact {
        url,
        dll_name,
        format,
        cache_subdir,
    })
}

/// Ensure the ORT DLL for `device` is present under `cache_dir` and return
/// its canonical path.
///
/// Returns the cached path on a hit, the freshly-extracted path on a
/// miss, or an error describing why the download failed (404, timeout,
/// extraction error, etc.). The caller propagates the error directly —
/// no silent fallback — so an operator shipping a wrong `ORT_VERSION`
/// sees the underlying HTTP status instead of a confusing ort load
/// failure.
///
/// # Concurrency
///
/// Two concurrent first-use callers (e.g. the watcher + the HTTP
/// extractor racing on a cold cache) coordinate through a sentinel file
/// `<cache_dir>/.<subdir>-download`: the winner (whose `O_CREAT | O_EXCL`
/// succeeds) downloads + extracts; the loser polls for the
/// `.<dll>-complete` marker. Mirrors `crate::nli::model_cache::PartClaim`
/// but for the directory-shaped ORT cache. Without this gate, both
/// callers would download the multi-hundred-MB archive and the loser's
/// staging-rename would clobber the winner's, racing on extraction
/// integrity.
pub async fn ensure_ort_binary(device: DeviceKind, cache_dir: &Path) -> Result<PathBuf> {
    let artifact = artifact_for(device)?;
    install_from_url(
        &artifact.url,
        artifact.dll_name,
        artifact.format,
        artifact.cache_subdir,
        cache_dir,
    )
    .await
}

/// Same as [`ensure_ort_binary`] but takes the archive URL explicitly.
///
/// Public-by-convention (still `pub(crate)`) so an integration test can
/// point at a wiremock fixture instead of `github.com`. The single-flight
/// claim + poll logic lives here so the test exercises the real race
/// protection end-to-end (count downloads via the mock server).
pub(crate) async fn install_from_url(
    url: &str,
    dll_name: &'static str,
    format: ArchiveFormat,
    cache_subdir: &'static str,
    cache_dir: &Path,
) -> Result<PathBuf> {
    let dll_dir = cache_dir.join(cache_subdir);
    let dll_path = dll_dir.join(dll_name);
    let complete_marker = dll_dir.join(format!(".{dll_name}-complete"));

    if dll_path.exists() && complete_marker.exists() {
        tracing::debug!(path = %dll_path.display(), "ORT DLL already cached");
        return Ok(dll_path);
    }

    std::fs::create_dir_all(cache_dir)?;
    let claim_path = cache_dir.join(format!(".{cache_subdir}-download"));

    match DownloadClaim::try_claim(claim_path.clone()) {
        Ok(claim) => {
            // Sole downloader. The claim is held until the function
            // returns so the sentinel stays alive across the (slow)
            // download + extraction; its `Drop` cleans up on either
            // success (so a later cache invalidation can re-claim) or
            // failure (so the next caller wins a fresh claim).
            let install_result = install_winner(
                url,
                dll_name,
                format,
                &dll_dir,
                &dll_path,
                &complete_marker,
            )
            .await;
            // Drop `claim` explicitly before returning so the sentinel
            // disappears the moment the install finishes — the loser's
            // poll loop keys off `complete_marker`, not the sentinel,
            // but freeing the sentinel eagerly lets a concurrent
            // re-invocation on a *different* device proceed.
            drop(claim);
            install_result
        }
        Err(ClaimError::AlreadyInProgress) => {
            tracing::info!(
                cache_subdir,
                "another caller is downloading ORT; polling cache for completion"
            );
            poll_for_completion(&dll_path, &complete_marker).await
        }
        Err(ClaimError::Io(e)) => Err(anyhow::Error::from(e)),
    }
}

/// Winner path: download → extract → atomic rename → completion marker.
///
/// Extracted so the claim branch reads as a single delegating call; the
/// function owns every filesystem mutation that must happen strictly
/// after winning the claim and strictly before the marker is written.
async fn install_winner(
    url: &str,
    dll_name: &'static str,
    format: ArchiveFormat,
    dll_dir: &Path,
    dll_path: &Path,
    complete_marker: &Path,
) -> Result<PathBuf> {
    // A partial prior extraction (DLL present but marker absent) is
    // treated as a cache miss and re-downloaded. Cheap because the next
    // start after a successful extraction always has the marker.
    if dll_dir.exists() && !complete_marker.exists() {
        tracing::warn!(
            dir = %dll_dir.display(),
            "removing partial ORT extraction left by a previous crashed run"
        );
        let _ = std::fs::remove_dir_all(dll_dir);
    }

    tracing::info!(url = %url, "downloading ONNX Runtime");
    let bytes = download_archive(url).await?;

    std::fs::create_dir_all(dll_dir)?;
    let staging_dir = unique_staging_dir(dll_dir);
    std::fs::create_dir_all(&staging_dir)?;

    let staging_dir_owned = staging_dir.clone();
    let _extracted = tokio::task::spawn_blocking(move || {
        extract_archive(&bytes, format, dll_name, &staging_dir_owned)
    })
    .await
    .map_err(|e| anyhow::anyhow!("extraction worker join error: {e}"))??;

    // Atomically promote the staging dir into place. The rename of the
    // directory itself is atomic on POSIX; on Windows `rename` over an
    // existing dir fails, but we already removed a partial `dll_dir`
    // above so the target is absent.
    if dll_dir.exists() {
        let _ = std::fs::remove_dir_all(dll_dir);
    }
    std::fs::rename(&staging_dir, dll_dir)?;

    // Marker is the very last write so its presence is proof the
    // directory reached its final, fully-populated state.
    std::fs::write(complete_marker, [])?;
    tracing::info!(path = %dll_path.display(), "ONNX Runtime ready");
    // Return the canonical post-rename path, NOT the staging path: the
    // staging directory was just renamed into `dll_dir`, so any path
    // computed before the rename points at a location that no longer
    // exists. The pre-B6 code returned the stale staging path, which
    // happened to work only because the caller (ort session init) opens
    // the canonical name itself.
    Ok(dll_path.to_path_buf())
}

/// Loser path: wait for the winner's `complete_marker` to appear.
///
/// Bounded — the loser must not block forever if the winner crashed
/// before writing the marker. The budget mirrors the per-request
/// download timeout ([`DOWNLOAD_TIMEOUT`]) plus generous slack for
/// extraction; on expiry the caller surfaces an error rather than
/// wedging the audit startup.
async fn poll_for_completion(dll_path: &Path, complete_marker: &Path) -> Result<PathBuf> {
    const RETRIES: u32 = 600;
    const RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);
    for _ in 0..RETRIES {
        if dll_path.exists() && complete_marker.exists() {
            return Ok(dll_path.to_path_buf());
        }
        tokio::time::sleep(RETRY_INTERVAL).await;
    }
    anyhow::bail!(
        "ORT cache did not materialize after {} retries (another caller holds the download claim): {}",
        RETRIES,
        dll_path.display()
    )
}

/// Single-flight claim on a sentinel file next to the ORT cache subdir.
///
/// Mirrors [`crate::nli::model_cache::PartClaim`] but guards the
/// directory-shaped ORT extraction (the model_cache version guards a
/// single-file copy). The claim is the `O_CREAT | O_EXCL` race on
/// `claim_path`: exactly one concurrent caller wins, the rest fall
/// through to the loser-poll branch.
struct DownloadClaim {
    path: PathBuf,
}

#[derive(Debug)]
enum ClaimError {
    /// Another caller won the `create_new` race and is mid-download.
    AlreadyInProgress,
    /// A real IO error (permission denied, read-only filesystem, …).
    Io(std::io::Error),
}

impl DownloadClaim {
    fn try_claim(path: PathBuf) -> Result<Self, ClaimError> {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => Ok(Self { path }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(ClaimError::AlreadyInProgress)
            }
            Err(e) => Err(ClaimError::Io(e)),
        }
    }
}

impl Drop for DownloadClaim {
    fn drop(&mut self) {
        // Best-effort cleanup so a later cold-cache miss can re-claim.
        // Errors are swallowed — nothing useful to do at drop time.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Download an archive with a bounded timeout and a hard size cap.
///
/// The size cap is enforced by reading the body in chunks and aborting
/// the moment the running total exceeds [`MAX_DOWNLOAD_BYTES`]; this
/// prevents a malicious or misconfigured CDN from streaming an
/// unbounded response into memory.
async fn download_archive(url: &str) -> Result<Vec<u8>> {
    use futures::StreamExt;

    let client = crate::upstream::http_client::with_timeout(DOWNLOAD_TIMEOUT)?;
    let response = client.get(url).send().await?.error_for_status()?;

    let mut buffer: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if buffer.len() + chunk.len() > MAX_DOWNLOAD_BYTES {
            anyhow::bail!(
                "download exceeded {} byte cap before completion (url: {url})",
                MAX_DOWNLOAD_BYTES
            );
        }
        buffer.extend_from_slice(&chunk);
    }
    Ok(buffer)
}

/// Synchronous extraction core. Runs on the blocking pool because the
/// `zip` and `tar` crates are inherently synchronous and a CUDA archive
/// can be hundreds of megabytes — running this on a tokio worker thread
/// would block every other future sharing that worker for seconds.
///
/// Extracts every shared library (not just the canonical DLL) into
/// `out_dir`: CUDA archives bundle `onnxruntime_providers_*.dll` and the
/// cuDNN runtime next to `onnxruntime.dll`, and the DirectML `.nupkg`
/// ships `DirectML.dll` alongside the ort runtime. All of them are
/// needed at session-build time and ort resolves them via the DLL's own
/// directory.
fn extract_archive(
    bytes: &[u8],
    format: ArchiveFormat,
    dll_name: &str,
    out_dir: &Path,
) -> Result<PathBuf> {
    match format {
        ArchiveFormat::Zip => extract_zip(bytes, dll_name, out_dir),
        ArchiveFormat::Tgz => extract_tgz(bytes, dll_name, out_dir),
    }
}

fn extract_zip(bytes: &[u8], dll_name: &str, out_dir: &Path) -> Result<PathBuf> {
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let mut found: Option<PathBuf> = None;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let entry = file.name().to_string();
        let basename = basename_of(&entry);
        if !is_shared_lib(basename) {
            continue;
        }
        let out_path = out_dir.join(basename);
        let mut out_file = std::fs::File::create(&out_path)?;
        std::io::copy(&mut file, &mut out_file)?;
        if basename == dll_name {
            found = Some(out_path);
        }
    }
    found.ok_or_else(|| anyhow::anyhow!("DLL `{dll_name}` not found in zip archive"))
}

/// `tar` + `flate2` decoder path. Same extraction contract as
/// [`extract_zip`]; separated only because the decoder types differ.
fn extract_tgz(bytes: &[u8], dll_name: &str, out_dir: &Path) -> Result<PathBuf> {
    let decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);

    let mut found: Option<PathBuf> = None;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let entry_path = entry.path()?.to_string_lossy().into_owned();
        let basename = basename_of(&entry_path);
        if !is_shared_lib(basename) {
            continue;
        }
        let out_path = out_dir.join(basename);
        entry.unpack(&out_path)?;
        if basename == dll_name {
            found = Some(out_path);
        }
    }
    found.ok_or_else(|| anyhow::anyhow!("DLL `{dll_name}` not found in tgz archive"))
}

/// Build a unique staging directory next to `target` so concurrent
/// startup attempts (rare, but possible if the operator accidentally
/// starts two `smos serve` instances on a cold cache) do not collide.
fn unique_staging_dir(target: &Path) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    target.with_file_name(format!(
        "{}.staging-{pid}-{nanos}",
        target.file_name().and_then(|s| s.to_str()).unwrap_or("ort")
    ))
}

/// Leaf segment of a `/`- or `\`-separated archive path.
fn basename_of(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// `true` if `basename` looks like a loadable shared library across the
/// three platforms SMOS supports. Conservative on purpose: matches
/// `*.dll`, `*.dylib`, and `lib*.so*` so we pull CUDA's `libcu*.so`
/// companions and DirectML's `DirectML.dll` while skipping import
/// libraries (`.lib`), PDBs, headers, and metadata.
fn is_shared_lib(basename: &str) -> bool {
    let lower = basename.to_ascii_lowercase();
    if lower.ends_with(".dll") || lower.ends_with(".dylib") {
        return true;
    }
    lower.starts_with("lib") && lower.contains(".so")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_shared_lib_matches_canonical_runtime_names() {
        assert!(is_shared_lib("onnxruntime.dll"));
        assert!(is_shared_lib("onnxruntime_providers_cuda.dll"));
        assert!(is_shared_lib("DirectML.dll"));
        assert!(is_shared_lib("libonnxruntime.so"));
        assert!(is_shared_lib("libonnxruntime.so.1.24.0"));
        assert!(is_shared_lib("libonnxruntime.dylib"));
        assert!(is_shared_lib("libcudnn.so.8"));
    }

    #[test]
    fn is_shared_lib_rejects_non_runtime_artifacts() {
        assert!(!is_shared_lib("onnxruntime.lib"));
        assert!(!is_shared_lib("onnxruntime.pdb"));
        assert!(!is_shared_lib("onnxruntime.h"));
        assert!(!is_shared_lib("README.md"));
        assert!(!is_shared_lib("[Content_Types].xml"));
    }

    #[test]
    fn basename_of_strips_directory_prefixes() {
        assert_eq!(basename_of("foo/bar/onnxruntime.dll"), "onnxruntime.dll");
        assert_eq!(basename_of("onnxruntime.dll"), "onnxruntime.dll");
        assert_eq!(basename_of("a\\b\\c.dll"), "c.dll");
        assert_eq!(
            basename_of("onnxruntime-win-x64-1.24.1/lib/onnxruntime.dll"),
            "onnxruntime.dll"
        );
    }

    #[test]
    fn archive_format_extension_round_trips() {
        assert_eq!(ArchiveFormat::Zip.extension(), "zip");
        assert_eq!(ArchiveFormat::Tgz.extension(), "tgz");
    }

    #[test]
    fn current_platform_returns_supported_descriptor() {
        // The dev/CI host MUST be one of the supported (os, arch) pairs;
        // an unsupported host would never have managed to build the
        // crate, so we can assert success here. Intel macOS is the one
        // supported-at-build-time target that is rejected at this layer
        // (no upstream ORT artifact), so it has its own test below.
        let (platform, dll, fmt) = current_platform().expect("current platform supported");
        assert!(matches!(platform, "win-x64" | "linux-x64" | "osx-arm64"));
        assert!(dll.starts_with("onnxruntime") || dll.starts_with("libonnxruntime"));
        assert!(fmt.extension() == "zip" || fmt.extension() == "tgz");
    }

    #[test]
    fn artifact_for_cpu_carries_version_and_subdir() {
        let a = artifact_for(DeviceKind::Cpu).expect("cpu artifact for current platform");
        assert!(
            a.url.contains(ORT_VERSION),
            "URL missing version: {}",
            a.url
        );
        assert_eq!(a.cache_subdir, "cpu");
        assert!(a.url.contains("github.com/microsoft/onnxruntime"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn artifact_for_metal_reuses_cpu_macos_build() {
        let a = artifact_for(DeviceKind::Metal).expect("metal artifact");
        assert_eq!(a.cache_subdir, "metal");
        // The Metal/CoreML EP is bundled in the CPU ort macOS build, so
        // the URL must NOT carry any `-gpu`/`-directml` infix.
        assert!(!a.url.contains("-gpu"));
        assert!(!a.url.contains("directml"));
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    #[test]
    fn artifact_for_directml_targets_nuget_v3_flat_container() {
        // DirectML is no longer a github release zip; verify the URL
        // points at the NuGet v3 flat-container API so a future
        // "helpful" refactor does not silently regress to a 404.
        let a = artifact_for(DeviceKind::DirectML).expect("directml artifact");
        assert_eq!(a.cache_subdir, "directml");
        assert!(
            a.url.starts_with(
                "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/"
            ),
            "DirectML URL must be the v3 flat-container endpoint, got: {}",
            a.url
        );
        assert!(a.url.ends_with(".nupkg"));
    }

    #[test]
    fn artifact_for_unsupported_device_returns_error() {
        // DirectML on non-Windows hosts is genuinely unsupported; the
        // detection layer never produces this combo in practice, but
        // an operator who hard-codes `device = "directml"` on Linux
        // needs a clear error rather than a silent CPU fallthrough.
        #[cfg(not(target_os = "windows"))]
        {
            let res = artifact_for(DeviceKind::DirectML);
            assert!(res.is_err(), "DirectML must error on non-Windows");
        }
        // Cuda on macOS is unsupported by upstream ORT.
        #[cfg(target_os = "macos")]
        {
            let res = artifact_for(DeviceKind::Cuda);
            assert!(res.is_err(), "CUDA must error on macOS");
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    #[test]
    fn current_platform_rejects_intel_macos() {
        // Apple Intel Macs were dropped from upstream ORT starting with
        // 1.24.x. The error message is the only signal an operator on
        // such a host gets, so it must mention the platform and the
        // workaround (pin ORT_VERSION to 1.23.x).
        let err = current_platform().expect_err("Intel macOS must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("osx-x86_64"),
            "error must name the dropped platform: {msg}"
        );
        assert!(
            msg.contains("1.23"),
            "error must mention the pin workaround: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // B6: single-flight download claim — exactly one download per cold cache
    // -----------------------------------------------------------------------
    //
    // The pre-B6 path raced two concurrent `ensure_ort_binary` callers on a
    // cold cache: both missed the existence check, both downloaded, and the
    // loser's staging-rename clobbered the winner's extraction. The B6 fix
    // gates the download behind a sentinel file (`O_CREAT | O_EXCL`); the
    // loser polls the completion marker instead of re-downloading. The two
    // tests below pin the contract via the extracted `install_from_url`
    // seam so they can point at a wiremock fixture instead of `github.com`.

    /// Build an in-memory zip archive containing a single fake `onnxruntime
    /// .dll` entry. Mirrors the layout Microsoft ships so `extract_archive`
    /// finds the canonical DLL name and the install path succeeds.
    fn fake_ort_zip() -> Vec<u8> {
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("onnxruntime.dll", opts).unwrap();
            zip.write_all(b"fake-ort-bytes").unwrap();
            zip.finish().unwrap();
        }
        buf.into_inner()
    }

    // Regression (B6): two concurrent installs against the same cold cache
    // MUST trigger exactly one HTTP download. The wiremock `MockServer`
    // records every received request; the assertion reads the server's
    // request log so the count is exact (not just "at least one"). The
    // loser wins the marker poll and returns the SAME canonical path the
    // winner installed, proving no double-extraction clobbered the result.
    #[tokio::test]
    async fn concurrent_installs_trigger_exactly_one_download() {
        use tempfile::TempDir;

        let body = fake_ort_zip();
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/ort.zip"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let _cache_keepalive = TempDir::new().expect("tempdir");
        let cache_dir = _cache_keepalive.path().to_path_buf();
        let url = format!("{}/ort.zip", server.uri());

        // Race two concurrent installs against the same cache_dir. The
        // claim sentinel (`.<subdir>-download`) serializes them: one wins
        // `O_CREAT | O_EXCL`, the other polls the completion marker.
        let url_a = url.clone();
        let url_b = url.clone();
        let cache_a = cache_dir.clone();
        let cache_b = cache_dir.clone();
        let (res_a, res_b) = tokio::join!(
            async move {
                install_from_url(
                    &url_a,
                    "onnxruntime.dll",
                    ArchiveFormat::Zip,
                    "test-device",
                    &cache_a,
                )
                .await
            },
            async move {
                install_from_url(
                    &url_b,
                    "onnxruntime.dll",
                    ArchiveFormat::Zip,
                    "test-device",
                    &cache_b,
                )
                .await
            },
        );

        let path_a = res_a.expect("winner install succeeded");
        let path_b = res_b.expect("loser install succeeded (via poll)");
        assert_eq!(
            path_a, path_b,
            "both concurrent callers must resolve to the SAME canonical path"
        );
        assert!(path_a.exists(), "canonical DLL must land on disk");
        assert!(
            path_a.ends_with("onnxruntime.dll"),
            "unexpected path: {}",
            path_a.display()
        );

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "exactly one HTTP download must fire (got {}); the second caller must poll, not re-download",
            received.len()
        );
    }

    // Regression (B6): a serial second install after the first has finished
    // must NOT re-download — the cache hit short-circuits before the claim.
    // Pins the fast-path so the claim + poll machinery is not invoked on a
    // warm cache (which would otherwise add a pointless 500 ms poll cycle).
    #[tokio::test]
    async fn warm_cache_hit_skips_download_entirely() {
        use tempfile::TempDir;

        let body = fake_ort_zip();
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/ort.zip"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .mount(&server)
            .await;

        let _cache_keepalive = TempDir::new().expect("tempdir");
        let cache_dir = _cache_keepalive.path().to_path_buf();
        let url = format!("{}/ort.zip", server.uri());

        let first = install_from_url(
            &url,
            "onnxruntime.dll",
            ArchiveFormat::Zip,
            "test-device-warm",
            &cache_dir,
        )
        .await
        .expect("first install");
        let second = install_from_url(
            &url,
            "onnxruntime.dll",
            ArchiveFormat::Zip,
            "test-device-warm",
            &cache_dir,
        )
        .await
        .expect("warm cache hit");
        assert_eq!(first, second);

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "warm-cache hit must NOT re-download (got {} requests)",
            received.len()
        );
    }
}
