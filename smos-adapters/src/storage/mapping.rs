use smos_application::errors::RepoError;
use smos_domain::{FactStatus, FactType, Timestamp};
use time::OffsetDateTime;

pub(crate) fn domain_to_repo(e: smos_domain::DomainError) -> RepoError {
    RepoError::SerializationFailed(e.to_string())
}

pub(crate) fn parse_fact_type(s: &str) -> Result<FactType, RepoError> {
    match s {
        "decision" => Ok(FactType::Decision),
        "preference" => Ok(FactType::Preference),
        "entity" => Ok(FactType::Entity),
        "event" => Ok(FactType::Event),
        "technical" => Ok(FactType::Technical),
        other => Err(RepoError::SerializationFailed(format!(
            "unknown fact_type: {other}"
        ))),
    }
}

pub(crate) fn parse_fact_status(s: &str) -> Result<FactStatus, RepoError> {
    match s {
        "pending" => Ok(FactStatus::Pending),
        "accepted" => Ok(FactStatus::Accepted),
        "rejected" => Ok(FactStatus::Rejected),
        other => Err(RepoError::SerializationFailed(format!(
            "unknown status: {other}"
        ))),
    }
}

pub(crate) fn format_iso(ts: OffsetDateTime) -> String {
    // `time`'s `Rfc3339` format is widely compatible and accepted by
    // SurrealDB's `datetime` parser. Formatting a `Rfc3339`-compatible
    // `OffsetDateTime` should never fail in practice, but the previous
    // silent fallback to `"1970-01-01T00:00:00Z"` would corrupt the
    // timestamp-dependent heat decay if it ever did. Surface the error at
    // ERROR level and fall back to the debug representation instead of
    // silently emitting the epoch.
    //
    // Caveat: the `time` crate's `Debug` for `OffsetDateTime` is NOT a
    // valid Rfc3339 string, so `parse_iso` will likely fail on a
    // round-trip. The branch is "should be unreachable" — the goal is to
    // avoid the silent epoch-corruption of heat decay timestamps and to
    // leave a forensic trail in the ERROR log + stored row, not to keep
    // the row readable.
    match ts.format(&time::format_description::well_known::Rfc3339) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(
                error = %e,
                timestamp = ?ts,
                "Rfc3339 formatting failed; this should be unreachable — please report"
            );
            format!("{:?}", ts)
        }
    }
}

pub(crate) fn parse_iso(s: &str) -> Result<Timestamp, RepoError> {
    OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
        .map_err(|e| RepoError::SerializationFailed(format!("invalid datetime {s:?}: {e}")))
        .and_then(|odt| {
            Timestamp::from_unix_secs(odt.unix_timestamp())
                .map_err(|e| RepoError::SerializationFailed(format!("unix out of range: {e}")))
        })
}
