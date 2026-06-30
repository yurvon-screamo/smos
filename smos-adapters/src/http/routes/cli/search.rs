//! `POST /v1/cli/search` — forwarded `smos search` execution.
//!
//! Invokes the SAME [`RetrieveFacts`] use case the CLI local branch invokes,
//! then renders through [`crate::cli::search_runner::render_json`]. The
//! response body is the final stdout document; the CLI remote branch pipes
//! it verbatim — no JSON round-trip on the client side.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use smos_application::ports::Clock as ClockPort;
use smos_application::use_cases::RetrieveFacts;

use crate::cli::import_helpers::parse_memory_key;
use crate::cli::search_runner::render_json;
use crate::http::axum_server::AppState;
use crate::http::error_mapper;
use crate::{LlamaCppReranker, OllamaEmbedding};

/// Wire body for `POST /v1/cli/search`. Mirrors
/// [`crate::cli::search_runner::SearchRequest`] on the producer side; kept
/// independent (Deserialize here, Serialize there) so the handler does not
/// depend on the CLI module's serialisation concerns.
#[derive(Debug, Deserialize)]
pub struct SearchRequestBody {
    pub query: String,
    pub memory_key: String,
    pub top_k: Option<usize>,
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequestBody>,
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

    let embedder = match OllamaEmbedding::new(Arc::new(state.config.embedding.clone())) {
        Ok(e) => e,
        Err(e) => {
            return error_mapper::error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!("embedder build failed: {e:#}"),
            );
        }
    };
    let reranker = match LlamaCppReranker::new(Arc::new(state.config.reranker.clone())) {
        Ok(r) => r,
        Err(e) => {
            return error_mapper::error_response(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!("reranker build failed: {e:#}"),
            );
        }
    };

    let clock = FlatClock(state.clock.clone());
    let retrieval_cfg = state.retrieval_cfg.clone();
    let heat_cfg = state.heat_cfg.clone();

    let use_case = RetrieveFacts {
        facts: &state.store,
        embedder: &embedder,
        reranker: &reranker,
        clock: &clock,
        retrieval_cfg: &retrieval_cfg,
        heat_cfg: &heat_cfg,
    };

    let scored = match use_case.execute(&req.query, &memory_key, req.top_k).await {
        Ok(s) => s,
        Err(error) => return error_mapper::render_use_case_error(error),
    };

    let body = render_json(&scored);
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    (axum::http::StatusCode::OK, headers, Bytes::from(body)).into_response()
}

/// Wrapper around `Arc<dyn Clock>` so the by-value `C: Clock` bound on
/// [`RetrieveFacts`] is satisfied. Mirrors the same-named wrapper in
/// [`crate::http::routes::chat_completions`]: both wrap the shared trait
/// object stored in [`AppState`] so handlers do not pay an `Arc<dyn>`→trait-
/// object indirection on the use case. Cheap to clone (one `Arc` bump).
#[derive(Clone)]
struct FlatClock(Arc<dyn ClockPort + Send + Sync>);

impl ClockPort for FlatClock {
    fn now(&self) -> smos_domain::Timestamp {
        self.0.now()
    }
}
