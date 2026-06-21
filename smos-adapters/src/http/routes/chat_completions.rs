//! `POST /v1/chat/completions` — OpenAI-compatible chat-completion handler.
//!
//! Pipeline (Slice-4 + Slice-5):
//! 1. Hand the request to `HandleChatCompletion`, which:
//!    - parses `"memory_key:real_model"` and strips the prefix,
//!    - detects / mints the session id from history,
//!    - runs `EnrichRequest` (memory retrieval + injection, fail-open),
//!    - forwards to the upstream.
//! 2. Inject the session marker into the upstream response.
//!    - Streaming → tunnel chunks 1:1 with the marker appended to the
//!      terminal `finish_reason="stop"` chunk.
//!    - Non-streaming → inject the marker into `choices[0].message.content`.
//! 3. Slice-5: spawn the background fact-extraction task AFTER the response.
//!    - Streaming → the stream wrapper finalizes a `StreamingBuffer` once
//!      `[DONE]` is reached and hands it to the spawner (non-blocking).
//!    - Non-streaming → the spawner runs immediately (the body is already
//!      complete).
//!    - When `enable_response_extraction = false` extraction is skipped
//!      entirely: the streaming path falls back to the lightweight marker-only
//!      wrapper (no per-chunk buffering), the non-streaming path skips the
//!      spawn outright.
//!
//! Extraction tasks are tracked by the [`ExtractionSupervisor`] so the server
//! can drain them on shutdown (`shutdown_extraction_grace_seconds`).
//!
//! The handler is intentionally thin: every piece of business logic lives in
//! the use case. HTTP-specific concerns (status codes, SSE framing, body
//! shapes) stay here.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use smos_application::ports::{Clock as ClockPort, IdGenerator as IdGeneratorPort};
use smos_application::types::{ChatRequest, ChatResponse};
use smos_application::use_cases::extract_facts_from_response::ExtractFactsFromResponse;
use smos_application::use_cases::{HandleChatCompletion, extract_response_payload};
use smos_domain::chat::ToolCall;
use smos_domain::config::ConfidenceConfig;
use smos_domain::config::ExtractionConfig;
use smos_domain::{MemoryKey, SessionId};

use crate::SurrealStore;
use crate::http::axum_server::AppState;
use crate::http::error_mapper::{error_response, render_use_case_error};
use crate::http::stream_transform::{self, ExtractionSpawner};
use crate::providers::{OllamaEmbedding, OllamaExtractor};
use crate::runtime::{ExtractionSupervisor, TokioDelay};
use crate::upstream::sse_parser;
use crate::upstream::streaming_buffer::StreamingBuffer;

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    let is_streaming = request.is_streaming();
    let enable_extraction = state.config.server.enable_response_extraction;

    let use_case = HandleChatCompletion {
        facts: state.store.clone(),
        sessions: state.store.clone(),
        embedder: state.embedder.clone(),
        reranker: state.reranker.clone(),
        upstream: state.upstream.clone(),
        clock: FlatClock(state.clock.clone()),
        id_generator: FlatIdGenerator(state.id_generator.clone()),
        retrieval_cfg: state.retrieval_cfg.clone(),
        heat_cfg: state.heat_cfg.clone(),
    };

    let (response, session_id, memory_key) = match use_case.execute(request).await {
        Ok(triple) => triple,
        Err(error) => return render_use_case_error(error),
    };

    let marker = session_id.to_marker();

    if is_streaming {
        streaming_response(
            state,
            response,
            marker,
            memory_key,
            session_id,
            enable_extraction,
        )
    } else {
        non_streaming_response(
            state,
            response,
            marker,
            memory_key,
            session_id,
            enable_extraction,
        )
    }
}

/// Build the SSE response. When extraction is ENABLED, the stream is wrapped
/// with a `StreamingBuffer` + extraction spawner; when DISABLED, it uses the
/// lightweight marker-only wrapper (no per-chunk buffering overhead). A
/// non-streaming upstream reply when streaming was requested is a protocol
/// mismatch → 500.
#[allow(clippy::too_many_arguments)]
fn streaming_response(
    state: Arc<AppState>,
    response: ChatResponse,
    marker: String,
    memory_key: MemoryKey,
    session_id: SessionId,
    enable_extraction: bool,
) -> Response {
    let stream = match response {
        ChatResponse::Streaming(s) => s,
        ChatResponse::NonStreaming(_) => {
            return error_response(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "upstream returned a buffered reply for a streaming request",
            );
        }
    };
    if enable_extraction {
        let spawner = ResponseExtractionSpawner::new(state, memory_key, session_id);
        let marked = stream_transform::inject_marker_with_extraction(
            stream,
            marker,
            StreamingBuffer::new(),
            spawner,
        );
        axum::response::sse::Sse::new(marked).into_response()
    } else {
        // Kill-switch off: skip the per-chunk buffer entirely.
        let marked = stream_transform::inject_marker(stream, marker);
        axum::response::sse::Sse::new(marked).into_response()
    }
}

/// Inject the marker into the buffered JSON reply, then spawn the extraction
/// task with the pre-marker content. A streaming reply when a buffered one was
/// requested is a protocol mismatch → 500.
#[allow(clippy::too_many_arguments)]
fn non_streaming_response(
    state: Arc<AppState>,
    response: ChatResponse,
    marker: String,
    memory_key: MemoryKey,
    session_id: SessionId,
    enable_extraction: bool,
) -> Response {
    match response {
        ChatResponse::NonStreaming(value) => {
            if enable_extraction {
                // Extract the payload BEFORE injecting the marker so the
                // extraction input never includes SMOS control noise.
                let (content, tool_calls) = extract_response_payload(&value);
                let spawner = ResponseExtractionSpawner::new(state, memory_key, session_id);
                spawner.spawn_extraction(content, tool_calls);
            }
            Json(sse_parser::inject_marker_non_streaming(value, &marker)).into_response()
        }
        ChatResponse::Streaming(_) => error_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "upstream returned a stream for a non-streaming request",
        ),
    }
}

/// Concrete extraction spawner owning every port the background task needs.
/// Cheap to build (all fields are `Arc`/clone-backed); consumed once by
/// `spawn_extraction`, which hands the task to the [`ExtractionSupervisor`]
/// so it survives a graceful shutdown.
struct ResponseExtractionSpawner {
    facts: SurrealStore,
    sessions: SurrealStore,
    embedder: OllamaEmbedding,
    extractor: OllamaExtractor,
    clock: FlatClock,
    delay: TokioDelay,
    confidence_cfg: Arc<ConfidenceConfig>,
    extraction_cfg: Arc<ExtractionConfig>,
    supervisor: ExtractionSupervisor,
    memory_key: MemoryKey,
    session_id: SessionId,
}

impl ResponseExtractionSpawner {
    fn new(state: Arc<AppState>, memory_key: MemoryKey, session_id: SessionId) -> Self {
        Self {
            facts: state.store.clone(),
            sessions: state.store.clone(),
            embedder: state.embedder.clone(),
            extractor: state.extractor.clone(),
            clock: FlatClock(state.clock.clone()),
            delay: TokioDelay,
            confidence_cfg: state.confidence_cfg.clone(),
            extraction_cfg: state.extraction_cfg.clone(),
            supervisor: state.extraction_supervisor.clone(),
            memory_key,
            session_id,
        }
    }
}

impl ExtractionSpawner for ResponseExtractionSpawner {
    fn spawn_extraction(self, content: String, tool_calls: Vec<ToolCall>) {
        let ResponseExtractionSpawner {
            facts,
            sessions,
            embedder,
            extractor,
            clock,
            delay,
            confidence_cfg,
            extraction_cfg,
            supervisor,
            memory_key,
            session_id,
        } = self;
        supervisor.spawn(async move {
            let use_case = ExtractFactsFromResponse {
                facts: &facts,
                sessions: &sessions,
                embedder: &embedder,
                extractor: &extractor,
                clock: &clock,
                delay: &delay,
                confidence_cfg: &confidence_cfg,
                extraction_cfg: &extraction_cfg,
                enable_response_extraction: true,
            };
            match use_case
                .execute(&content, &tool_calls, &memory_key, &session_id)
                .await
            {
                Ok(count) => tracing::info!(
                    count,
                    session = %session_id,
                    "background response extraction completed"
                ),
                Err(error) => tracing::warn!(
                    %error,
                    session = %session_id,
                    "background response extraction failed (non-blocking)"
                ),
            }
        });
    }
}

/// Wrapper around `Arc<dyn Clock>` that implements `Clock` by delegating.
///
/// `HandleChatCompletion` requires `C: Clock` (a by-value bound), but the
/// shared state holds the clock behind a trait object. This newtype forwards
/// calls and is cheap to clone (one `Arc` bump).
#[derive(Clone)]
struct FlatClock(Arc<dyn ClockPort + Send + Sync>);

impl ClockPort for FlatClock {
    fn now(&self) -> smos_domain::Timestamp {
        self.0.now()
    }
}

/// Wrapper around `Arc<dyn IdGenerator>` that implements `IdGenerator` by
/// delegating. Same shape as [`FlatClock`]: the shared state holds the id
/// generator behind a trait object, but `HandleChatCompletion` requires
/// `IG: IdGenerator` (a by-value bound).
#[derive(Clone)]
struct FlatIdGenerator(Arc<dyn IdGeneratorPort + Send + Sync>);

impl IdGeneratorPort for FlatIdGenerator {
    fn new_session_id(&self) -> SessionId {
        self.0.new_session_id()
    }
}
