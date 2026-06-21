//! Map `UpstreamError` to HTTP responses (§12 error matrix).
//!
//! A single [`ErrorClass`] classification drives both the status code and the
//! body shape, so the "which status" and "how to render" decisions stay in one
//! place (no duplicated range checks). Classes:
//! - `Verbatim(status)` — upstream 4xx: forward the upstream's own body so
//!   OpenAI clients that parse `error.code`/`error.type` keep working. The
//!   content-type is sniffed from the body (JSON vs text/plain) since the
//!   upstream's header is not preserved on the error path.
//! - `BadGateway` — upstream unreachable / timed out / 5xx / unparseable: 502
//!   with a generic message (upstream internal details never leak).
//! - `Internal` — SMOS-side request encoding bug: 500 with the cause.

use axum::Json;
use axum::body::Bytes;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use smos_application::errors::{UpstreamError, UseCaseError};

/// The single source of truth for how an `UpstreamError` is classified.
#[derive(Debug, PartialEq, Eq)]
enum ErrorClass {
    /// Forward the upstream's status and body verbatim.
    Verbatim(u16),
    /// Upstream failure — 502, generic message, no body leak.
    BadGateway,
    /// SMOS-side bug — 500, surface the cause.
    Internal,
}

fn classify(error: &UpstreamError) -> ErrorClass {
    match error {
        UpstreamError::StatusError { status, .. } if (400..500).contains(status) => {
            ErrorClass::Verbatim(*status)
        }
        UpstreamError::StatusError { .. }
        | UpstreamError::ConnectFailed(_)
        | UpstreamError::Timeout(_)
        | UpstreamError::StreamError(_)
        | UpstreamError::BadResponse(_)
        | UpstreamError::AllProvidersFailed(_) => ErrorClass::BadGateway,
        UpstreamError::SerializationError(_) => ErrorClass::Internal,
    }
}

/// Convert an `UpstreamError` into the HTTP response that best matches §12.
pub fn render(error: UpstreamError) -> Response {
    match classify(&error) {
        ErrorClass::Verbatim(status) => propagate_verbatim(status, error_body(&error)),
        ErrorClass::BadGateway => error_response(StatusCode::BAD_GATEWAY, message_for(&error)),
        ErrorClass::Internal => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, message_for(&error))
        }
    }
}

/// Convert a `UseCaseError` into an HTTP response.
///
/// - `Upstream` variant delegates to [`render`] (preserves the §12 status
///   matrix for upstream failures).
/// - `Domain` variant maps to 400 Bad Request: in Slice-4 the only path for a
///   domain error to reach the HTTP boundary is `parse_model` validating the
///   user-supplied model string. If a future slice surfaces an invariant
///   violation (status transition, confidence threshold), the handler will
///   treat it as a client error too — those never happen from well-formed
///   inputs in production.
/// - `Repo` / `Provider` variants map to 503 Service Unavailable: those are
///   recoverable downstream outages (DB down, Ollama down). The client can
///   retry; the proxy stays up.
pub fn render_use_case_error(error: UseCaseError) -> Response {
    match error {
        UseCaseError::Upstream(upstream) => render(upstream),
        UseCaseError::Domain(domain) => error_response(
            StatusCode::BAD_REQUEST,
            format!("invalid request: {domain}"),
        ),
        UseCaseError::Repo(repo) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("SMOS storage unavailable: {repo}"),
        ),
        UseCaseError::Provider(provider) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("SMOS provider unavailable: {provider}"),
        ),
    }
}

/// Build an OpenAI-shaped JSON error response body with a status code.
pub fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let body = Json(json!({
        "error": {
            "message": message.into(),
            "type": "smos_proxy_error",
        }
    }));
    (status, body).into_response()
}

/// The status code an `UpstreamError` maps to (kept public for tests/tools).
pub fn status_for(error: &UpstreamError) -> StatusCode {
    match classify(error) {
        ErrorClass::Verbatim(status) => {
            StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY)
        }
        ErrorClass::BadGateway => StatusCode::BAD_GATEWAY,
        ErrorClass::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// The client-facing message for an `UpstreamError`. For upstream failures we
/// emit a generic reason (never the upstream's body, which may carry internal
/// details); for SMOS-side serialization we surface the cause.
fn message_for(error: &UpstreamError) -> String {
    match error {
        UpstreamError::ConnectFailed(_) => "upstream LLM is unreachable".to_string(),
        UpstreamError::Timeout(_) => "upstream LLM request timed out".to_string(),
        UpstreamError::StreamError(_) => "upstream LLM stream was interrupted".to_string(),
        UpstreamError::BadResponse(_) => {
            "upstream LLM returned an unparseable response".to_string()
        }
        UpstreamError::StatusError { status, .. } => {
            format!("upstream LLM returned HTTP {status}")
        }
        UpstreamError::AllProvidersFailed(_) => {
            "every configured upstream LLM provider failed".to_string()
        }
        UpstreamError::SerializationError(_) => error.to_string(),
    }
}

/// The upstream body to forward verbatim (empty string for non-`StatusError`).
fn error_body(error: &UpstreamError) -> String {
    match error {
        UpstreamError::StatusError { body, .. } => body.clone(),
        _ => String::new(),
    }
}

/// Forward an upstream response body verbatim, sniffing a sane content-type
/// from the body itself (JSON-ish → `application/json`, otherwise
/// `text/plain`). The upstream's own content-type header is not carried on the
/// error path, so this avoids an `application/json` header over an HTML/text
/// body (gateway error pages, plain-text 4xx from proxies).
fn propagate_verbatim(status: u16, body: String) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST);
    let mut headers = HeaderMap::new();
    let trimmed = body.trim_start();
    let is_json = trimmed.starts_with('{') || trimmed.starts_with('[');
    let content_type = if is_json {
        "application/json"
    } else {
        "text/plain; charset=utf-8"
    };
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_str(content_type).expect("static content-type is valid ascii"),
    );
    (code, headers, Bytes::from(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn connect_failed_classifies_as_bad_gateway() {
        assert_eq!(
            classify(&UpstreamError::ConnectFailed("x".into())),
            ErrorClass::BadGateway
        );
    }

    #[test]
    fn timeout_classifies_as_bad_gateway() {
        assert_eq!(
            classify(&UpstreamError::Timeout(Duration::from_secs(5))),
            ErrorClass::BadGateway
        );
    }

    #[test]
    fn bad_response_classifies_as_bad_gateway() {
        assert_eq!(
            classify(&UpstreamError::BadResponse("not json".into())),
            ErrorClass::BadGateway
        );
    }

    #[test]
    fn upstream_4xx_classifies_as_verbatim() {
        let err = UpstreamError::StatusError {
            status: 422,
            body: "{}".into(),
        };
        assert_eq!(classify(&err), ErrorClass::Verbatim(422));
        assert_eq!(status_for(&err), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn upstream_5xx_classifies_as_bad_gateway_without_body_leak() {
        let err = UpstreamError::StatusError {
            status: 500,
            body: "INTERNAL: stacktrace at /etc/secrets".into(),
        };
        assert_eq!(classify(&err), ErrorClass::BadGateway);
        assert_eq!(status_for(&err), StatusCode::BAD_GATEWAY);
        assert!(!message_for(&err).contains("stacktrace"));
        assert!(!message_for(&err).contains("/etc/secrets"));
    }

    #[test]
    fn serialization_classifies_as_internal() {
        assert_eq!(
            classify(&UpstreamError::SerializationError("s".into())),
            ErrorClass::Internal
        );
        assert_eq!(
            status_for(&UpstreamError::SerializationError("s".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
