//! `POST /v1/chat/completions` — OpenAI-compatible chat-completion handler.
//!
//! Pipeline (Slice-4 + Slice-5):
//! 1. Hand the request to `HandleChatCompletion`, which:
//!    - parses `"memory_key:real_model"` and strips the prefix,
//!    - detects / mints the session id from history,
//!    - runs `EnrichRequest` (memory retrieval + injection). Enrichment is
//!      **fail-open** for embedder / vector-search / dedup (forwards the
//!      original messages) and **fail-closed** for the reranker (provider
//!      error or empty result → `UseCaseError::Provider(_)` → HTTP 503),
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

    // Extract the last user message BEFORE the request is consumed by the use
    // case. It threads into the background extraction task as framing context
    // for the assistant reply. One String clone per request — negligible.
    let user_message = last_user_text(&request.messages);

    // The routing maps are pre-built once at startup (see AppState) and
    // cloned via Arc here — no per-request HashMap/Vec allocation. If live
    // config reload is added later, the Arc will need to be swapped
    // atomically (e.g. via `ArcSwap`).
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
        persons: state.persons_view.clone(),
        providers: state.providers_view.clone(),
    };

    let (response, session_id, memory_key) = match use_case.execute(request).await {
        Ok(triple) => triple,
        Err(error) => return render_use_case_error(error),
    };

    let marker = session_id.to_marker();

    let ctx = ResponseContext {
        state,
        response,
        marker,
        user_message,
        memory_key,
        session_id,
        enable_extraction,
    };
    if is_streaming {
        streaming_response(ctx)
    } else {
        non_streaming_response(ctx)
    }
}

/// Shared inputs for the streaming / non-streaming response builders.
///
/// Groups the positional params the two builders previously took so the
/// `handle` dispatch and the builders read by field name. The fields mirror
/// the previous parameter list verbatim (same names, same types, same order)
/// plus `user_message` threaded into the extraction task.
struct ResponseContext {
    state: Arc<AppState>,
    response: ChatResponse,
    marker: String,
    user_message: String,
    memory_key: MemoryKey,
    session_id: SessionId,
    enable_extraction: bool,
}

/// Return the trimmed text of the LAST message whose `role == "user"`.
///
/// Content shape:
/// - plain string → trimmed verbatim;
/// - multipart array → the `type:"text"` parts are joined with `\n`
///   (image/audio/binary parts are dropped — they carry no fact context and
///   their base64 payload would bloat the extraction input, defeating the
///   token-saving goal);
/// - array with no text parts (image-only), missing content, or any other
///   shape → empty string. The search does NOT fall back to an earlier user
///   message in that case: an image-only turn genuinely has no textual
///   question, and an earlier prompt would inject unrelated context.
///
/// Returns an empty string when no user message exists, so the extraction use
/// case falls back to its label-free path.
fn last_user_text(messages: &[serde_json::Value]) -> String {
    for msg in messages.iter().rev() {
        let is_user = msg
            .get("role")
            .and_then(serde_json::Value::as_str)
            .map(|r| r == "user")
            .unwrap_or(false);
        if !is_user {
            continue;
        }
        return extract_user_text(msg.get("content"));
    }
    String::new()
}

/// Extract the textual content of one user message: plain string verbatim, or
/// the `\n`-joined non-empty `type:"text"` parts of a multipart array.
/// Anything else (non-text parts, missing content, numbers, …) yields "".
fn extract_user_text(content: Option<&serde_json::Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    if let Some(text) = content.as_str() {
        return text.trim().to_string();
    }
    let Some(parts) = content.as_array() else {
        return String::new();
    };
    parts
        .iter()
        .filter_map(|part| {
            let is_text = part
                .get("type")
                .and_then(serde_json::Value::as_str)
                .map(|t| t == "text")
                .unwrap_or(false);
            if !is_text {
                return None;
            }
            part.get("text")
                .and_then(serde_json::Value::as_str)
                .filter(|t| !t.is_empty())
        })
        .collect::<Vec<&str>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Build the SSE response. When extraction is ENABLED, the stream is wrapped
/// with a `StreamingBuffer` + extraction spawner; when DISABLED, it uses the
/// lightweight marker-only wrapper (no per-chunk buffering overhead). A
/// non-streaming upstream reply when streaming was requested is a protocol
/// mismatch → 500.
fn streaming_response(ctx: ResponseContext) -> Response {
    let ResponseContext {
        state,
        response,
        marker,
        user_message,
        memory_key,
        session_id,
        enable_extraction,
    } = ctx;
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
            user_message,
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
fn non_streaming_response(ctx: ResponseContext) -> Response {
    let ResponseContext {
        state,
        response,
        marker,
        user_message,
        memory_key,
        session_id,
        enable_extraction,
    } = ctx;
    match response {
        ChatResponse::NonStreaming(value) => {
            if enable_extraction {
                // Extract the payload BEFORE injecting the marker so the
                // extraction input never includes SMOS control noise.
                let (content, tool_calls) = extract_response_payload(&value);
                let spawner = ResponseExtractionSpawner::new(state, memory_key, session_id);
                spawner.spawn_extraction(user_message, content, tool_calls);
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
    fn spawn_extraction(self, user_message: String, content: String, tool_calls: Vec<ToolCall>) {
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
                .execute(
                    &user_message,
                    &content,
                    &tool_calls,
                    &memory_key,
                    &session_id,
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user(content: serde_json::Value) -> serde_json::Value {
        json!({"role": "user", "content": content})
    }

    #[test]
    fn last_user_text_returns_trimmed_string_content() {
        let msgs = [user(json!("  hello world  "))];
        assert_eq!(last_user_text(&msgs), "hello world");
    }

    #[test]
    fn last_user_text_picks_last_user_message_when_many() {
        let msgs = [
            user(json!("first question")),
            json!({"role": "assistant", "content": "answer"}),
            user(json!("second question")),
        ];
        assert_eq!(last_user_text(&msgs), "second question");
    }

    #[test]
    fn last_user_text_extracts_text_part_and_drops_image_from_multipart() {
        let msgs = [user(json!([
            {"type": "text", "text": "describe the scaling policy"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
        ]))];
        assert_eq!(last_user_text(&msgs), "describe the scaling policy");
    }

    #[test]
    fn last_user_text_joins_multiple_text_parts_with_newline() {
        let msgs = [user(json!([
            {"type": "text", "text": "q1"},
            {"type": "text", "text": "q2"}
        ]))];
        assert_eq!(last_user_text(&msgs), "q1\nq2");
    }

    #[test]
    fn last_user_text_drops_empty_text_parts() {
        let msgs = [user(json!([
            {"type": "text", "text": ""},
            {"type": "text", "text": "real"}
        ]))];
        assert_eq!(last_user_text(&msgs), "real");
    }

    #[test]
    fn last_user_text_image_only_array_yields_empty() {
        let msgs = [user(json!([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
        ]))];
        assert_eq!(
            last_user_text(&msgs),
            "",
            "image-only turn carries no textual question"
        );
    }

    #[test]
    fn last_user_text_does_not_fall_back_to_earlier_user_message_on_image_only_turn() {
        // The LAST user turn is image-only. We must NOT return the earlier
        // text turn's content — that would inject an unrelated prompt.
        let msgs = [
            user(json!("earlier textual question")),
            user(json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}])),
        ];
        assert_eq!(
            last_user_text(&msgs),
            "",
            "image-only turn must not fall back to an earlier user message"
        );
    }

    #[test]
    fn last_user_text_no_user_message_yields_empty() {
        let msgs = [
            json!({"role": "system", "content": "you are helpful"}),
            json!({"role": "assistant", "content": "answer"}),
        ];
        assert_eq!(last_user_text(&msgs), "");
    }

    #[test]
    fn last_user_text_missing_content_field_yields_empty() {
        let msgs = [json!({"role": "user"})];
        assert_eq!(last_user_text(&msgs), "");
    }
}
