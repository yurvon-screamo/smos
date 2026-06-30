//! `POST /v1/cli/finalize` — forwarded `smos finalize` execution.
//!
//! Invokes the SAME `FinalizeSession` use case the CLI local branch
//! invokes (through [`run_finalize_pipeline`]), then renders through the
//! SAME [`print_finalize_report`]. The response body is the final stdout
//! document; the CLI remote branch pipes it verbatim.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use smos_domain::SessionId;

use crate::cli::finalize_runner::{print_finalize_report, run_finalize_pipeline};
use crate::http::axum_server::AppState;
use crate::http::error_mapper;
use smos_application::log_nonfatal;

/// Wire body for `POST /v1/cli/finalize`.
#[derive(Debug, Deserialize)]
pub struct FinalizeRequestBody {
    pub session_id: String,
    pub memory_key: Option<String>,
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FinalizeRequestBody>,
) -> Response {
    let session_id = match SessionId::from_raw(&req.session_id) {
        Ok(sid) => sid,
        Err(e) => {
            return error_mapper::error_response(
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid session_id: {e}"),
            );
        }
    };

    let classifier_arc = match &state.classifier {
        Some(c) => c.clone(),
        None => {
            return error_mapper::error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "NLI classifier not available (server started without a working NLI backend)",
            );
        }
    };
    let classifier: &crate::NativeNliClassifier = classifier_arc.as_ref();

    let (aggregated, keys_scanned) = match run_finalize_pipeline(
        &state.store,
        classifier,
        &state.config,
        &session_id,
        req.memory_key.as_deref(),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return error_mapper::error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("finalize pipeline failed: {e:#}"),
            );
        }
    };

    // Lazy git-sync: initialise the OnceCell on the first finalize request.
    // Once `None` is stored (empty repo_url or open failure) the cell is
    // permanently None for the process lifetime — fail-open is the right
    // trade-off: the finalize result is valid, the git export is a
    // best-effort side effect.
    let git_mgr: Option<Arc<tokio::sync::Mutex<crate::git_sync::GitSyncManager>>> = match state
        .git_sync
        .get_or_try_init(|| async {
            if state.config.git.repo_url.trim().is_empty() {
                return Ok::<_, std::convert::Infallible>(None);
            }
            match crate::git_sync::GitSyncManager::open_or_clone(&state.config.git) {
                Ok(mgr) => Ok(Some(Arc::new(tokio::sync::Mutex::new(mgr)))),
                Err(e) => {
                    tracing::warn!(
                        error = %format!("{e:#}"),
                        "git sync manager init failed; finalize result is still valid"
                    );
                    Ok(None)
                }
            }
        })
        .await
    {
        Ok(opt) => opt.clone(),
        Err(_) => None,
    };

    if let Some(mgr_arc) = git_mgr
        && !aggregated.memory_keys.is_empty()
    {
        let mgr = mgr_arc.lock().await;
        log_nonfatal!(
            crate::cli::finalize_runner::export_to_git(&mgr, &state.store, &aggregated.memory_keys)
                .await,
            "git sync export failed; finalize result is still valid"
        );
    }

    let body = print_finalize_report(&aggregated, keys_scanned);
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    (axum::http::StatusCode::OK, headers, Bytes::from(body)).into_response()
}
