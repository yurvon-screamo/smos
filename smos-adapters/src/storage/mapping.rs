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

// Formats a timestamp as an Rfc3339 string for storage. Formatting is fail-
// closed: a timestamp whose Rfc3339 representation cannot be produced (e.g.
// a negative year, which the `time` crate rejects with
// `InvalidComponent("year")`) surfaces as `RepoError::SerializationFailed`
// rather than falling back to a `Debug` string. The previous `Debug`
// fallback was NOT a valid Rfc3339 datetime, so `parse_iso` could not
// round-trip it — corrupting the heat-decay timestamps that depend on the
// stored value.
pub(crate) fn format_iso(ts: OffsetDateTime) -> Result<String, RepoError> {
    ts.format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| RepoError::SerializationFailed(format!("Rfc3339 formatting failed: {e}")))
}

pub(crate) fn parse_iso(s: &str) -> Result<Timestamp, RepoError> {
    OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
        .map_err(|e| RepoError::SerializationFailed(format!("invalid datetime {s:?}: {e}")))
        .and_then(|odt| {
            Timestamp::from_unix_secs(odt.unix_timestamp())
                .map_err(|e| RepoError::SerializationFailed(format!("unix out of range: {e}")))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::{Date, Month, PrimitiveDateTime, Time};

    // Regression (B1): a constructible `OffsetDateTime` whose Rfc3339 format
    // is rejected must fail-closed with `RepoError::SerializationFailed`
    // instead of returning a `Debug` fallback that breaks the
    // `parse_iso` round-trip and corrupts heat-decay timestamps.
    // A negative year (`time` rejects it as `InvalidComponent("year")`) is
    // such a value: `Date::from_calendar_date(-1, ..)` succeeds but Rfc3339
    // formatting does not.
    #[test]
    fn format_iso_returns_err_on_invalid_offsetdatetime() {
        let date =
            Date::from_calendar_date(-1, Month::January, 1).expect("year -1 is a valid Date");
        let time = Time::from_hms(0, 0, 0).unwrap();
        let ts = PrimitiveDateTime::new(date, time).assume_utc();

        let result = format_iso(ts);

        assert!(
            result.is_err(),
            "format_iso must fail-closed on an OffsetDateTime Rfc3339 rejects"
        );
        match result {
            Err(RepoError::SerializationFailed(_)) => {}
            other => panic!("expected SerializationFailed, got {other:?}"),
        }
    }
}
