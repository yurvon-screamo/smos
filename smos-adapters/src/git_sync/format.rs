//! Markdown frontmatter format for fact round-trip.
//!
//! Each fact is serialised as a markdown file with a TOML frontmatter block
//! delimited by `---` lines and the fact's textual content as the body:
//!
//! ```text
//! ---
//! id = "..."
//! memory_key = "..."
//! status = "accepted"
//! ...
//! ---
//!
//! Rust is memory-safe.
//! ```
//!
//! The TOML frontmatter carries every scalar the domain `Fact` needs to
//! round-trip back through [`smos_domain::Fact::rehydrate`], with the
//! notable exception of the embedding vector — embeddings are NOT serialised
//! to disk because (a) they are large, (b) they are model-specific, and
//! (c) the import path re-embeds via the configured embedding provider.

use serde::{Deserialize, Serialize};
use smos_domain::{
    Confidence, Fact, FactContent, FactId, FactRecord, FactStatus, FactType, Heat, MemoryKey,
    SourceSessions, Timestamp,
};
use time::OffsetDateTime;

/// Round-trippable projection of a [`Fact`] onto markdown frontmatter.
///
/// Field order is significant: TOML table serialisation is insertion-order,
/// so listing the scalars here in the order an operator would scan them
/// (identity → lifecycle → provenance → time → heat) keeps the rendered
/// file readable when opened in a text editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactFrontmatter {
    pub id: String,
    pub memory_key: String,
    pub status: String,
    pub confidence: f32,
    pub fact_type: String,
    pub source_sessions: Vec<String>,
    pub conflicts_with: Vec<String>,
    pub extracted_at: String,
    pub valid_from: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
    pub heat_base: f32,
    pub last_access_at: String,
}

/// Render `fact` as a self-contained markdown string (frontmatter + body).
///
/// Returns `Err` rather than silently emitting an empty frontmatter block
/// when TOML serialisation fails — a partial write would produce a file
/// that round-trip-parses to `None` and silently drop the fact on re-import.
pub fn render_fact_md(fact: &Fact) -> Result<String, RenderError> {
    let fm = FactFrontmatter::from_fact(fact);
    let body = fact.content();
    let toml = toml::to_string(&fm).map_err(RenderError::Serialize)?;
    Ok(format!("---\n{toml}---\n\n{body}\n"))
}

/// Error raised by [`render_fact_md`].
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("toml serialization failed: {0}")]
    Serialize(#[from] toml::ser::Error),
}

/// Parse a markdown file produced by [`render_fact_md`] back into
/// `(frontmatter, body)`. Returns `None` on any structural failure so the
/// import path can skip the file with a warning rather than aborting the
/// whole import.
pub fn parse_fact_md(content: &str) -> Option<(FactFrontmatter, String)> {
    let trimmed = content.trim_start_matches('\u{feff}');
    let after_open = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))?;
    let fm_end = find_frontmatter_close(after_open)?;
    let (fm_block, rest) = after_open.split_at(fm_end);
    let fm: FactFrontmatter = toml::from_str(fm_block).ok()?;
    let body = strip_leading_delimiter(rest).trim().to_string();
    Some((fm, body))
}

/// Locate the byte offset of the closing `---` delimiter within the
/// frontmatter body. The delimiter must be at the start of a line; `---`
/// inside a TOML string value would otherwise be a false positive.
fn find_frontmatter_close(s: &str) -> Option<usize> {
    let mut cursor = 0;
    for line in s.split_inclusive('\n') {
        let trimmed_end = line.trim_end_matches(['\n', '\r']);
        if trimmed_end == "---" {
            return Some(cursor);
        }
        cursor += line.len();
    }
    None
}

/// Drop the closing `---` delimiter line and any leading blank lines so
/// the body starts at the actual content.
fn strip_leading_delimiter(rest: &str) -> &str {
    let without_delim = rest
        .strip_prefix("---\n")
        .or_else(|| rest.strip_prefix("---\r\n"))
        .unwrap_or(rest);
    without_delim.trim_start_matches(['\n', '\r'])
}

impl FactFrontmatter {
    /// Project a [`Fact`] into its serialisable form.
    pub fn from_fact(fact: &Fact) -> Self {
        Self {
            id: fact.id().as_str().to_string(),
            memory_key: fact.memory_key().as_str().to_string(),
            status: fact.status().as_str().to_string(),
            confidence: fact.confidence().value(),
            fact_type: fact.fact_type().as_str().to_string(),
            source_sessions: fact
                .source_sessions()
                .iter()
                .map(|s| s.as_str().to_string())
                .collect(),
            conflicts_with: fact
                .conflicts_with()
                .iter()
                .map(|f| f.as_str().to_string())
                .collect(),
            extracted_at: format_iso(fact.extracted_at().as_offset_date_time()),
            valid_from: format_iso(fact.valid_from().as_offset_date_time()),
            valid_until: fact
                .valid_until()
                .map(|t| format_iso(t.as_offset_date_time())),
            heat_base: fact.heat_base().value(),
            last_access_at: format_iso(fact.last_access_at().as_offset_date_time()),
        }
    }

    /// Rebuild the domain object via [`Fact::rehydrate`]. Embeddings are NOT
    /// stored in markdown; the caller is expected to re-embed via the
    /// configured embedding provider after import.
    pub fn to_fact(&self, body: &str) -> Result<Fact, RehydrateError> {
        let id =
            FactId::from_raw(&self.id).map_err(|e| RehydrateError::invalid("id", e.to_string()))?;
        let memory_key = MemoryKey::from_raw(&self.memory_key)
            .map_err(|e| RehydrateError::invalid("memory_key", e.to_string()))?;
        let content = FactContent::new(body.to_string())
            .map_err(|e| RehydrateError::invalid("content", e.to_string()))?;
        let fact_type = parse_fact_type(&self.fact_type);
        let confidence = Confidence::new(self.confidence)
            .map_err(|e| RehydrateError::invalid("confidence", e.to_string()))?;
        let status = parse_status(&self.status)?;
        let valid_from = parse_ts(&self.valid_from, "valid_from")?;
        let extracted_at = parse_ts(&self.extracted_at, "extracted_at")?;
        let valid_until = match &self.valid_until {
            Some(raw) => Some(parse_ts(raw, "valid_until")?),
            None => None,
        };
        let last_access_at = parse_ts(&self.last_access_at, "last_access_at")?;
        let heat_base = Heat::new(self.heat_base)
            .map_err(|e| RehydrateError::invalid("heat_base", e.to_string()))?;
        let source_sessions = self
            .source_sessions
            .iter()
            .map(|s| {
                smos_domain::SessionId::from_raw(s)
                    .map_err(|e| RehydrateError::invalid("source_sessions", e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let conflicts_with = self
            .conflicts_with
            .iter()
            .map(|s| {
                FactId::from_raw(s)
                    .map_err(|e| RehydrateError::invalid("conflicts_with", e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Fact::rehydrate(FactRecord {
            id,
            memory_key,
            content,
            fact_type,
            confidence,
            status,
            valid_from,
            valid_until,
            extracted_at,
            source_sessions: SourceSessions::from_vec(source_sessions),
            conflicts_with,
            heat_base,
            last_access_at,
            embedding: None,
        })
        .map_err(|e| RehydrateError::invalid("rehydrate", e.to_string()))
    }
}

/// Map a frontmatter `fact_type` string into the domain enum. Unknown
/// values log a WARN and fall back to `Entity` rather than aborting the
/// rehydrate — a single typo in a fact file must not block the entire
/// import.
fn parse_fact_type(raw: &str) -> FactType {
    match raw {
        "decision" => FactType::Decision,
        "preference" => FactType::Preference,
        "entity" => FactType::Entity,
        "event" => FactType::Event,
        "technical" => FactType::Technical,
        other => {
            tracing::warn!(fact_type = other, "unknown fact_type, defaulting to Entity");
            FactType::Entity
        }
    }
}

/// Errors raised by [`FactFrontmatter::to_fact`]. Carries a single string
/// so callers can log a one-line warning per skipped file.
#[derive(Debug, thiserror::Error)]
#[error("frontmatter rehydrate failed: {0}")]
pub struct RehydrateError(String);

impl RehydrateError {
    fn invalid(field: &str, detail: impl Into<String>) -> Self {
        Self(format!("{field}: {}", detail.into()))
    }
}

fn parse_status(s: &str) -> Result<FactStatus, RehydrateError> {
    match s {
        "pending" => Ok(FactStatus::Pending),
        "accepted" => Ok(FactStatus::Accepted),
        "rejected" => Ok(FactStatus::Rejected),
        other => Err(RehydrateError::invalid(
            "status",
            format!("unknown {other:?}"),
        )),
    }
}

fn parse_ts(raw: &str, field: &'static str) -> Result<Timestamp, RehydrateError> {
    let odt = OffsetDateTime::parse(raw, &time::format_description::well_known::Rfc3339)
        .map_err(|e| RehydrateError::invalid(field, format!("{raw:?}: {e}")))?;
    Timestamp::from_unix_secs(odt.unix_timestamp())
        .map_err(|e| RehydrateError::invalid(field, e.to_string()))
}

/// Rfc3339 formatting. Mirrors the `format_iso` helper in
/// `surreal_store` so markdown round-trips use the same wire format as the
/// SurrealDB rows; kept local to avoid widening the adapter's public API
/// with a date-formatting helper.
fn format_iso(ts: OffsetDateTime) -> String {
    match ts.format(&time::format_description::well_known::Rfc3339) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(
                error = %e,
                timestamp = ?ts,
                "Rfc3339 formatting failed; this should be unreachable — please report"
            );
            format!("{ts:?}")
        }
    }
}
