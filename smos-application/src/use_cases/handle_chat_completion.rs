//! `HandleChatCompletion` — top-level chat-completion use case (§3 + §4).
//!
//! Orchestrates the full request-side pipeline:
//! 1. Resolve the requested person via [`route_request`] and rewrite
//!    `request.model` to the upstream model id. The person name becomes the
//!    [`MemoryKey`] used by enrichment + extraction.
//! 2. Inject the persona `.md` (if any) as the system message so the
//!    upstream model reads it first.
//! 3. Detect the session id from the trailing 20 messages' markers (or mint
//!    a fresh one).
//! 4. Run [`EnrichRequest`] (memory retrieval + injection). Fail-open for
//!    every port EXCEPT the reranker: an embedder / vector-search / dedup
//!    failure forwards the original messages unchanged, but a reranker
//!    failure (provider error or empty result) propagates as
//!    `Err(UseCaseError::Provider(_))` so the HTTP handler returns 503.
//!    See [`EnrichRequest`] for the rationale.
//! 5. Forward the (possibly enriched) request to the LLM upstream, naming
//!    the provider that the person routes to.
//!
//! Slice-5 extraction is wired in the **adapter** layer (`http/`), not here.
//! The application layer stays runtime-agnostic: `tokio::spawn` requires a
//! multi-thread runtime, and the SMOS codebase keeps every runtime operation
//! (spawn, serve, signal handling) inside `smos`. The adapter wraps
//! the response stream with a `StreamingBuffer`, and after `[DONE]` spawns the
//! [`ExtractFactsFromResponse`] use case. This use case hands the adapter the
//! `MemoryKey` it needs for that wiring.
//!
//! Returns `(ChatResponse, SessionId, MemoryKey)` so the HTTP handler injects
//! the session marker AND the adapter wires extraction with the right project.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use smos_domain::chat::{ToolArguments, ToolCall};
use smos_domain::config::{HeatConfig, RetrievalConfig};
use smos_domain::{MemoryKey, SessionId};

use crate::errors::UseCaseError;
use crate::helpers::person_router::{
    PersonEntry, ProviderEntry, inject_persona_into_messages, load_persona_at, route_request,
};
use crate::helpers::session_marker;
use crate::ports::{
    Clock, EmbeddingProvider, FactRepository, IdGenerator, LlmUpstream, RerankProvider,
    SessionRepository,
};
use crate::types::{ChatRequest, ChatResponse, enrichment_messages_from_json};
use crate::use_cases::enrich_request::EnrichRequest;

/// Top-level chat-completion orchestration.
///
/// Owns the ports the REQUEST-side pipeline needs (enrichment + upstream
/// forwarding) plus the routing maps (`persons`, `providers`) needed to
/// resolve a requested person into a concrete upstream route. Extraction
/// ports live in `AppState` and are wired by the adapter — see the module
/// docs for the layering rationale.
pub struct HandleChatCompletion<FR, SR, EP, RP, LU, C, IG> {
    pub facts: FR,
    pub sessions: SR,
    pub embedder: EP,
    pub reranker: RP,
    pub upstream: LU,
    pub clock: C,
    pub id_generator: IG,
    pub retrieval_cfg: Arc<RetrievalConfig>,
    pub heat_cfg: Arc<HeatConfig>,
    pub persons: Arc<HashMap<String, PersonEntry>>,
    pub providers: Arc<Vec<ProviderEntry>>,
}

impl<FR, SR, EP, RP, LU, C, IG> HandleChatCompletion<FR, SR, EP, RP, LU, C, IG>
where
    FR: FactRepository,
    SR: SessionRepository,
    EP: EmbeddingProvider,
    RP: RerankProvider,
    LU: LlmUpstream,
    C: Clock,
    IG: IdGenerator,
{
    /// Run the chat-completion pipeline.
    ///
    /// Returns the upstream response, the session id (so the handler injects
    /// the marker), and the memory namespace (so the adapter spawns the
    /// extraction task against the correct project).
    pub async fn execute(
        &self,
        mut request: ChatRequest,
    ) -> Result<(ChatResponse, SessionId, MemoryKey), UseCaseError> {
        // Step 1 — route the requested person to a (memory_key, provider,
        // upstream_model, persona_path) tuple.
        let route = route_request(&request.model, &self.persons, &self.providers)?;
        let memory_key = route.memory_key;
        request.model = route.upstream_model;
        let provider_name = route.provider_name;

        // Step 2 — inject the persona as the leading system message when the
        // person declares one. The persona file is loaded synchronously
        // here because: (a) persona `.md` files are tiny (<1 KB typical),
        // (b) after the first access the page cache serves subsequent
        // reads in microseconds, (c) an LLM proxy's request latency is
        // dominated by the upstream round-trip (seconds), so the
        // one-time cold-cache read on the first request is noise. The
        // fail-soft contract (`None` on missing/unreadable file) preserves
        // the proxy's overall availability story.
        if let Some(persona_path) = route.persona_path
            && let Some(persona_content) = load_persona_at(&persona_path)
        {
            inject_persona_into_messages(&mut request.messages, &persona_content);
        }

        // H-5: build a *read-only* typed projection of the messages for
        // the helpers that need to introspect message content (session
        // marker detection here, topic extraction inside EnrichRequest).
        // The projection is NEVER re-serialised back into
        // `request.messages` — the raw `Vec<Value>` stays the source of
        // truth so per-message fields the typed DTO does not model
        // (`name`, `tool_call_id`, `refusal`, `image_url` parts, audio
        // parts, future OpenAI extensions) survive the enrichment
        // pipeline verbatim. The previous round-trip design lost those
        // fields and broke the fail-open contract for tool-calling and
        // vision workflows.
        let typed_projection = enrichment_messages_from_json(&request.messages);

        // Step 3 — detect session from the typed projection. Falls back
        // to a freshly-minted id from the injected generator when no
        // marker survived in the trailing window. Generation goes
        // through the `IdGenerator` port so the domain's
        // `SessionId::new()` constructor stays `pub(crate)` and id
        // generation is an explicit, mockable capability.
        let session_id = session_marker::detect_from_typed_messages(&typed_projection)
            .unwrap_or_else(|| self.id_generator.new_session_id());

        // Step 4 — enrichment. Fail-open for embedder / vector-search / dedup
        // (those return the original messages on `Ok`); fail-closed for the
        // reranker (propagates as `Err(UseCaseError::Provider(_))` → HTTP
        // 503). `std::mem::take` is safe because every `Ok` path returns at
        // least the original messages — no `Vec::new()` replacement risk.
        // The `?` propagates ONLY the reranker error to the handler; every
        // other failure already fail-opened inside `execute`.
        let enriched_messages = self
            .enrich(
                std::mem::take(&mut request.messages),
                &memory_key,
                &session_id,
            )
            .await?;
        request.messages = enriched_messages;

        // Step 5 — forward.
        let response = self.upstream.complete(&provider_name, request).await?;
        Ok((response, session_id, memory_key))
    }

    /// Run `EnrichRequest` and propagate its `Result`. The only `Err` path
    /// is the reranker ([`UseCaseError::Provider`]); every other port-level
    /// failure already fail-opened inside `execute` and returned the
    /// original messages. The wrapper exists only to keep `execute`
    /// readable.
    async fn enrich(
        &self,
        messages: Vec<Value>,
        memory_key: &MemoryKey,
        session_id: &SessionId,
    ) -> Result<Vec<Value>, UseCaseError> {
        let enrich = EnrichRequest {
            facts: &self.facts,
            sessions: &self.sessions,
            embedder: &self.embedder,
            reranker: &self.reranker,
            clock: &self.clock,
            retrieval_cfg: &self.retrieval_cfg,
            heat_cfg: &self.heat_cfg,
        };
        enrich.execute(messages, memory_key, session_id).await
    }
}

/// Extract the assistant content + structured tool calls from an OpenAI-shaped
/// non-streaming response so the extraction pipeline can reason over both.
///
/// `arguments` arrives as a JSON **string** on the wire (OpenAI quirk); the
/// domain stores it verbatim as an opaque [`ToolArguments`] — parsing is
/// deferred to the adapter layer that actually needs to interpret the
/// payload. Exported so the adapter can run the same parsing on the buffered
/// non-streaming body before spawning the background extraction task.
pub fn extract_response_payload(value: &Value) -> (String, Vec<ToolCall>) {
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let tool_calls = value
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(parse_openai_tool_call).collect())
        .unwrap_or_default();
    (content, tool_calls)
}

/// Convert one OpenAI tool-call object (`{id, type, function:{name, arguments}}`)
/// into the domain [`ToolCall`] shape.
///
/// `arguments` is normalised to a JSON-shaped string so the opaque
/// [`ToolArguments`] is always text — the OpenAI string form is forwarded
/// verbatim; an actual JSON object (some servers send that) is re-serialised;
/// a missing field degrades to `"null"`.
fn parse_openai_tool_call(v: &Value) -> Option<ToolCall> {
    let function = v.get("function")?;
    let name = function.get("name")?.as_str()?.to_string();
    let arguments = match function.get("arguments") {
        Some(Value::String(raw)) => raw.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
        None => "null".to_string(),
    };
    Some(ToolCall {
        name,
        arguments: ToolArguments::from_json(arguments),
    })
}
