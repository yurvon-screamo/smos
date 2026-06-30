//! `POST /v1/cli/import/raw` — forwarded `smos import raw` execution.
//!
//! Invokes the SAME extraction + optional finalize pipeline the CLI local
//! branch invokes (through [`run_raw_import_pipeline`]), then renders
//! through the SAME [`render_raw_import_report`]. The response body is the
//! final stdout document; the CLI remote branch pipes it verbatim.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::cli::import_helpers::{derive_session_id, parse_memory_key};
use crate::cli::raw_import_runner::{render_raw_import_report, run_raw_import_pipeline};
use crate::http::axum_server::AppState;
use crate::http::error_mapper;

/// Wire body for `POST /v1/cli/import/raw`.
#[derive(Debug, Deserialize)]
pub struct RawImportRequestBody {
    pub text: String,
    pub memory_key: String,
    pub no_finalize: bool,
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RawImportRequestBody>,
) -> Response {
    let memory_key = match parse_memory_key(&req.memory_key) {
        Ok(mk) => mk,
        Err(e) => {
            return error_mapper::error_response(
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid memory_key: {e}"),
            );
        }
    };

    // Pre-check: the operator asked for finalize (no_finalize=false) but the
    // server has no working NLI backend. Fail loud (503) BEFORE extraction
    // so the operator is not left with pending facts they expected to be
    // finalized. Consistent with the /v1/cli/finalize handler's contract.
    if !req.no_finalize && state.classifier.is_none() {
        return error_mapper::error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "NLI classifier not available: finalize was requested but the server \
             started without a working NLI backend. Re-run with --no-finalize or \
             restart the proxy after fixing the NLI model.",
        );
    }

    let session_id = derive_session_id("raw-import");
    let classifier = if req.no_finalize {
        None
    } else {
        state.classifier.as_ref().map(|arc| arc.as_ref())
    };

    let result =
        match run_raw_import_pipeline(crate::cli::raw_import_runner::RawImportPipelineRequest {
            store: &state.store,
            config: &state.config,
            text: &req.text,
            memory_key: &memory_key,
            session_id: &session_id,
            no_finalize: req.no_finalize,
            classifier,
        })
        .await
        {
            Ok(r) => r,
            Err(e) => {
                return error_mapper::error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("raw import pipeline failed: {e:#}"),
                );
            }
        };

    let body = render_raw_import_report(&result, req.no_finalize);
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    (axum::http::StatusCode::OK, headers, Bytes::from(body)).into_response()
}
