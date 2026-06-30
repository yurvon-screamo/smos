//! `POST /v1/cli/import/opencode` — forwarded `smos import opencode` execution.
//!
//! Invokes the SAME [`ImportOpencodeSession`] use case the CLI local branch
//! invokes (through [`run_import_opencode_pipeline`]), then renders through
//! the SAME [`render_import_opencode_report`]. The response body is the
//! postlude only (the prelude — `Source:` / `Parsed N turns` / `After
//! offset/limit` — was already printed CLI-side in BOTH branches).

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};

use crate::cli::import_helpers::parse_memory_key;
use crate::cli::import_runner::{
    ImportOpencodeRequest, render_import_opencode_report, run_import_opencode_pipeline,
};
use crate::http::axum_server::AppState;
use crate::http::error_mapper;

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ImportOpencodeRequest>,
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

    let turns: Vec<_> = req.turns.into_iter().map(Into::into).collect();

    let stats = match run_import_opencode_pipeline(
        crate::cli::import_runner::ImportOpencodePipelineRequest {
            store: &state.store,
            config: &state.config,
            turns,
            memory_key: &memory_key,
            session_id_str: &req.session_id,
            agents: &req.agents,
        },
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            return error_mapper::error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("import opencode pipeline failed: {e:#}"),
            );
        }
    };

    let body = render_import_opencode_report(&stats, &memory_key);
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    (axum::http::StatusCode::OK, headers, Bytes::from(body)).into_response()
}
