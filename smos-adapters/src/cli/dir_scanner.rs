//! Pure helpers for `smos import-dir`: recursive directory scan +
//! content-type-aware text extraction. All functions are sync and IO-light
//! so the unit tests in this module run without async runtime / Ollama.

use std::path::{Path, PathBuf};

use walkdir::{DirEntry, WalkDir};

/// File extensions the directory importer accepts. Lowercase, no leading dot.
/// `yml` is included alongside `yaml` because both are in common use.
const SUPPORTED_EXTENSIONS: &[&str] = &["md", "txt", "json", "jsonl", "yaml", "yml", "toml"];

/// Minimum length (bytes) for a JSON string value to be lifted into the
/// extraction input. Filters out structural noise (single-word keys, short
/// enum values like `"open"` / `"closed"`) that would otherwise crowd the
/// extractor input with non-factual boilerplate.
const MIN_JSON_VALUE_LEN: usize = 20;

/// Recursively walk `dir`, returning every supported file path in
/// deterministic (alphabetical) order. Hidden entries (any path component
/// starting with `.` — covers `.git`, `.svn`, `.idea`, …) are skipped so
/// VCS metadata never leaks into the import. walkdir errors (permission
/// denied, broken symlinks, symlink loops) are logged at WARN and the
/// offending entry is dropped — the rest of the tree still imports.
pub fn scan_directory(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in WalkDir::new(dir)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(is_visible)
    {
        match entry {
            Ok(entry) => {
                if entry.file_type().is_file() && is_supported_file(entry.path()) {
                    out.push(entry.into_path());
                }
            }
            Err(e) => tracing::warn!(error = %e, "skipping unreadable directory entry"),
        }
    }
    out
}

/// walkdir filter predicate: keep the root (depth 0) unconditionally so a
/// caller-supplied hidden root path still works, drop every other entry
/// whose file name starts with `.`. Returns `true` = keep, `false` =
/// prune (matches `WalkDir::filter_entry`'s contract).
fn is_visible(entry: &DirEntry) -> bool {
    entry.depth() == 0
        || entry
            .file_name()
            .to_str()
            .map(|name| !name.starts_with('.'))
            .unwrap_or(true)
}

/// Match a path's extension against [`SUPPORTED_EXTENSIONS`]
/// (case-insensitive). Files without an extension return `false`.
pub fn is_supported_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| {
            let lower = ext.to_ascii_lowercase();
            SUPPORTED_EXTENSIONS.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

/// Read `path` and lift its textual content into a single `String` suitable
/// for the extraction pipeline. The return type separates the two failure
/// modes that the runner must distinguish:
///
/// - `Err(io::Error)` — file unreadable (missing, non-UTF-8, permission
///   denied, …). Surfaces to the operator as "SKIP (read error: …)" so a
///   TOCTOU deletion or dangling symlink is not silently relabeled as
///   "no extractable content".
/// - `Ok(None)` — file read successfully but carries no qualifying text
///   (all-numeric JSON, an empty document, an all-keys-too-short object).
///   Surfaced as "SKIP (no extractable content)".
/// - `Ok(Some(text))` — text ready for the extraction pipeline.
///
/// For JSON / JSONL the extractor's structural string-extraction runs
/// here; for prose formats (`md`, `txt`, `yaml`, `yml`, `toml`) the raw
/// content is returned verbatim — YAML / TOML parsing would add a heavy
/// dependency for marginal gain over letting the LLM extractor read the
/// raw text directly.
pub fn read_file_content(path: &Path) -> Result<Option<String>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let extracted = match ext.as_str() {
        "json" => extract_text_from_json(&content),
        "jsonl" => extract_text_from_jsonl(&content),
        _ => Some(content),
    };
    Ok(extracted)
}

/// Parse one JSON document and concatenate every "long enough" string value
/// it contains (recursively, across arrays and objects). Returns `None`
/// when the document is unparseable or carries no qualifying strings.
fn extract_text_from_json(content: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let mut texts = Vec::new();
    collect_long_strings(&value, &mut texts);
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n\n"))
    }
}

/// Parse a JSONL document (one JSON value per non-empty line) and lift the
/// qualifying strings from every line into a single buffer. Parse failures
/// on individual lines are logged at WARN and skipped so one corrupt line
/// does not abort the whole file — but the operator sees the signal in the
/// log because corrupt source data is a quality issue worth surfacing.
fn extract_text_from_jsonl(content: &str) -> Option<String> {
    let mut texts = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(value) => collect_long_strings(&value, &mut texts),
            Err(e) => tracing::warn!(error = %e, "skipping unparseable JSONL line"),
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n\n"))
    }
}

/// Depth-first walk over a `serde_json::Value` tree that pushes every
/// string longer than [`MIN_JSON_VALUE_LEN`] into `out`. Numbers, bools,
/// and nulls are ignored — they are structural, not factual.
fn collect_long_strings(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => {
            if s.len() > MIN_JSON_VALUE_LEN {
                out.push(s.clone());
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                collect_long_strings(v, out);
            }
        }
        serde_json::Value::Object(obj) => {
            for (_, v) in obj {
                collect_long_strings(v, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    //! Pure-helper coverage: scanner ordering + hidden-dir skipping,
    //! extension matching, JSON / JSONL / raw text extraction. Filesystem
    //! fixtures live in `tempfile::tempdir()` so the tests are
    //! hermetic and OS-agnostic.

    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(dir: &Path, rel: &str, body: &str) -> PathBuf {
        let path = dir.join(rel);
        fs::create_dir_all(path.parent().expect("rel has a parent")).unwrap();
        fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn is_supported_file_matches_all_advertised_extensions() {
        for ext in ["md", "txt", "json", "jsonl", "yaml", "yml", "toml"] {
            let path = PathBuf::from(format!("doc.{ext}"));
            assert!(is_supported_file(&path), "expected .{ext} to be supported");
        }
    }

    #[test]
    fn is_supported_file_rejects_unknown_extensions() {
        for ext in ["rs", "go", "py", "", "PDF", "MD5"] {
            let path = PathBuf::from(if ext.is_empty() {
                "noext".to_string()
            } else {
                format!("file.{ext}")
            });
            assert!(
                !is_supported_file(&path),
                "expected .{ext:?} to be rejected"
            );
        }
    }

    #[test]
    fn is_supported_file_is_case_insensitive() {
        assert!(is_supported_file(&PathBuf::from("X.MD")));
        assert!(is_supported_file(&PathBuf::from("X.Json")));
    }

    #[test]
    fn scan_directory_returns_files_in_alphabetical_order() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write(root, "b.md", "b");
        write(root, "a.md", "a");
        write(root, "c.txt", "c");

        let files = scan_directory(root);
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.md", "b.md", "c.txt"]);
    }

    #[test]
    fn scan_directory_skips_hidden_directories_and_files() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write(root, "visible.md", "v");
        write(root, ".git/config.md", "hidden");
        write(root, ".cache/notes.md", "hidden");
        write(root, "sub/.hidden.md", "hidden");

        let files = scan_directory(root);
        assert_eq!(files.len(), 1, "only the visible file should remain");
        assert_eq!(
            files[0].file_name().unwrap().to_string_lossy(),
            "visible.md"
        );
    }

    #[test]
    fn scan_directory_descends_into_subdirectories() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write(root, "top.md", "t");
        write(root, "nested/inner.md", "i");
        write(root, "nested/deep/deep.md", "d");

        let files = scan_directory(root);
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn scan_directory_ignores_unsupported_extensions() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write(root, "ok.md", "m");
        write(root, "skip.rs", "fn main(){}");
        write(root, "skip.png", "binary");

        let files = scan_directory(root);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_name().unwrap().to_string_lossy(), "ok.md");
    }

    #[test]
    fn read_file_content_returns_raw_for_markdown_and_text() {
        let dir = tempdir().unwrap();
        let md = write(dir.path(), "a.md", "# Title\n\nbody text");
        let txt = write(dir.path(), "b.txt", "plain text body");

        assert_eq!(
            read_file_content(&md).unwrap().unwrap(),
            "# Title\n\nbody text"
        );
        assert_eq!(read_file_content(&txt).unwrap().unwrap(), "plain text body");
    }

    #[test]
    fn read_file_content_returns_raw_for_yaml_and_toml() {
        let dir = tempdir().unwrap();
        let yaml = write(dir.path(), "a.yaml", "key: value\nother: 1\n");
        let toml = write(dir.path(), "b.toml", "[section]\nkey = \"value\"\n");

        assert!(
            read_file_content(&yaml)
                .unwrap()
                .unwrap()
                .contains("key: value")
        );
        assert!(
            read_file_content(&toml)
                .unwrap()
                .unwrap()
                .contains("[section]")
        );
    }

    #[test]
    fn read_file_content_extracts_long_json_string_values() {
        let dir = tempdir().unwrap();
        let body = r#"{
            "title": "This title is long enough to qualify",
            "short": "x",
            "nested": { "note": "Another qualifying string value here" },
            "numbers": [1, 2, 3],
            "items": [{"body": "Item body that exceeds the minimum length"}]
        }"#;
        let path = write(dir.path(), "doc.json", body);

        let extracted = read_file_content(&path)
            .unwrap()
            .expect("JSON had qualifying strings");
        assert!(extracted.contains("This title is long enough to qualify"));
        assert!(extracted.contains("Another qualifying string value here"));
        assert!(extracted.contains("Item body that exceeds the minimum length"));
        assert!(!extracted.contains("short"));
    }

    #[test]
    fn read_file_content_returns_none_for_json_without_long_strings() {
        let dir = tempdir().unwrap();
        let body = r#"{"a": 1, "b": "short", "c": [true, false]}"#;
        let path = write(dir.path(), "doc.json", body);
        assert!(read_file_content(&path).unwrap().is_none());
    }

    #[test]
    fn read_file_content_returns_none_for_unparseable_json() {
        let dir = tempdir().unwrap();
        let path = write(dir.path(), "bad.json", "{not valid json");
        assert!(read_file_content(&path).unwrap().is_none());
    }

    #[test]
    fn read_file_content_handles_jsonl_with_multiple_lines() {
        let dir = tempdir().unwrap();
        let body = concat!(
            "{\"text\": \"first qualifying line of jsonl\"}\n",
            "\n",
            "{\"text\": \"second qualifying line of jsonl\"}\n",
            "{\"bad\": ",
            "garbage\n",
            "{\"num\": 42, \"body\": \"third qualifying line of jsonl\"}\n"
        );
        let path = write(dir.path(), "doc.jsonl", body);

        let extracted = read_file_content(&path)
            .unwrap()
            .expect("JSONL had qualifying lines");
        assert!(extracted.contains("first qualifying line of jsonl"));
        assert!(extracted.contains("second qualifying line of jsonl"));
        assert!(extracted.contains("third qualifying line of jsonl"));
    }

    #[test]
    fn read_file_content_surfaces_io_error_for_missing_file() {
        let path = PathBuf::from("/this/path/does/not/exist/smos-test.md");
        assert!(
            read_file_content(&path).is_err(),
            "missing file must surface as Err(io::Error), not Ok(None)"
        );
    }

    #[test]
    fn read_file_content_surfaces_io_error_for_non_utf8_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("binary.md");
        // 0xFF 0xFE is an invalid UTF-8 sequence (and the UTF-16 BOM without
        // the rest of the document); read_to_string must reject it.
        std::fs::write(&path, [0xFFu8, 0xFE, b' ', b'x']).unwrap();
        assert!(
            read_file_content(&path).is_err(),
            "non-UTF-8 file must surface as Err(io::Error)"
        );
    }
}
