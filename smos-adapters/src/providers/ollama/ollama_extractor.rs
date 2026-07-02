//! `OllamaExtractor` — `LlmExtractor` against an OpenAI-compatible
//! `/v1/chat/completions` endpoint backed by `llama-server` (Qwen3.5-2B-MTP
//! by default).
//!
//! Sends a system+user prompt pair (few-shot instructions + the response text),
//! parses the bullet-list reply, and filters prompt-echo noise so SMOS control
//! text never becomes a "fact". HTTP-level failures map to the
//! [`ProviderError`] shape the application retry loop expects:
//!
//! - connection refused / timeout → `Unavailable` (graceful skip, no retry).
//! - non-2xx status → `RequestFailed` (retried by the use case).
//! - malformed body → `InvalidResponse` (retried).
//!
//! The use case pre-combines content + formatted tool calls into the
//! `response_content` argument, so the adapter uses it verbatim and does not
//! re-format the (already-inlined) tool calls.

use std::sync::Arc;
use std::time::Instant;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use smos_application::errors::ProviderError;
use smos_application::ports::LlmExtractor;
use smos_domain::chat::ToolCall;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::config::LlmExtractionConfig;
use crate::providers::ollama::ollama_client::build_client;

/// System prompt: KNOWLEDGE-fact extraction contract (POC `prompts.py` shape).
///
/// Kept as one constant so prompt tweaks live in one place. The model is told
/// to preserve technical terms verbatim and to emit one fact per `- ` bullet.
const EXTRACTION_SYSTEM_PROMPT: &str = "\
Extract KNOWLEDGE facts from the text below.\n\
Each fact is a standalone English assertion capturing WHAT was learned, decided, or discovered — not HOW it was investigated.\n\
\n\
Preserve technical terms EXACTLY: file paths (auth.rs), code identifiers (validate_token), commands (cargo test), version numbers (TTL=60), proper nouns.\n\
Translate non-English content to English.\n\
\n\
DO extract:\n\
- Architecture decisions and component relationships\n\
- Stable technical facts (what something does, how something works)\n\
- Bug root causes and fixes applied\n\
- Configuration values and their effects\n\
- User preferences\n\
\n\
DO NOT extract:\n\
- Trivial actions ('User opened file auth.rs', 'Read file Cargo.toml')\n\
- Process noise ('cd /project && cargo build', 'ls -la')\n\
- Ephemeral state ('Currently in debugging session')\n\
- Meta-commentary, intentions, or restating the task\n\
\n\
Output as a bullet list, one fact per line starting with \"- \". Quality over quantity.";

/// OpenAI-compatible fact extractor backed by `llama-server`
/// (Qwen3.5-2B-MTP by default).
///
/// An optional [`ExtractionGate`] bounds how many concurrent `extract_facts`
/// HTTP calls may be in flight at once. The serve path injects one sized to
/// `llm_extraction.max_concurrent_extractions` so background extractions
/// cannot pile up on a shared single-slot upstream (`-np 1`) and starve
/// chat-completion forwards; the sequential CLI import paths leave it `None`.
#[derive(Clone)]
pub struct OllamaExtractor {
    client: Client,
    config: Arc<LlmExtractionConfig>,
    extraction_slot: Option<ExtractionGate>,
}

/// Concurrency gate around extraction HTTP calls. Owns both the semaphore
/// (the actual gate) AND its total permit count, so occupancy
/// (`permits − available_permits`) is derivable without coupling to
/// `LlmExtractionConfig`. The total is carried here — not read from config
/// at log time — so a caller that wires a semaphore of a different size
/// (e.g. a test) gets an accurate `concurrent` log field instead of one
/// computed against a stale config value.
#[derive(Clone)]
struct ExtractionGate {
    semaphore: Arc<Semaphore>,
    permits: usize,
}

impl ExtractionGate {
    fn held(&self) -> usize {
        self.permits
            .saturating_sub(self.semaphore.available_permits())
    }
}

impl OllamaExtractor {
    /// Build the adapter with a pooled HTTP client sized to the config timeout.
    /// Construction does NOT contact the server. No extraction-slot gate is
    /// wired: the sequential CLI import paths use this directly.
    pub fn new(config: Arc<LlmExtractionConfig>) -> Result<Self, ProviderError> {
        let client = build_client(config.timeout_seconds)?;
        Ok(Self {
            client,
            config,
            extraction_slot: None,
        })
    }

    /// Attach a concurrency gate around the extraction HTTP call. Consumes and
    /// returns `self` so the serve wiring chains it after [`new`]:
    ///
    /// ```ignore
    /// // `permits` MUST equal the semaphore's permit count so the gate's
    /// // occupancy log field reports true in-flight count.
    /// OllamaExtractor::new(cfg)?.with_slot(Arc::new(Semaphore::new(1)), 1)
    /// ```
    ///
    /// (Snippet is `ignore`d because it needs an `LlmExtractionConfig` and the
    /// crate's items in scope; see `tests/extraction_concurrency_gate.rs` for a
    /// runnable end-to-end example.)
    ///
    /// `permits` MUST equal the semaphore's permit count so the gate-contention
    /// log field (`concurrent = permits − available`) reports true occupancy.
    /// `extract_facts` acquires a permit before issuing the request and holds
    /// it for the call's duration (released on drop of the owned permit). The
    /// gate bounds concurrent in-flight extraction HTTP calls to the permit
    /// count, so background extractions cannot occupy every slot of a shared
    /// single-slot upstream and starve chat-completion forwards.
    pub fn with_slot(mut self, slot: Arc<Semaphore>, permits: usize) -> Self {
        self.extraction_slot = Some(ExtractionGate {
            semaphore: slot,
            permits,
        });
        self
    }

    fn chat_url(&self) -> String {
        format!(
            "{}/v1/chat/completions",
            self.config.url.trim_end_matches('/')
        )
    }

    /// Acquire a gate permit (when a gate is wired), recording the wait and the
    /// gate's true occupancy. Returns the owned permit (held across the HTTP
    /// call) plus the wait duration and post-acquire occupancy, or `None` when
    /// no gate is wired (CLI sequential paths) OR the semaphore was closed
    /// mid-shutdown: extraction is background work and must not wedge on a
    /// permit, so the caller proceeds ungated and logs the non-fatal miss.
    async fn acquire_gate(&self) -> Option<(OwnedSemaphorePermit, std::time::Duration, usize)> {
        let gate = self.extraction_slot.clone()?;
        let started = Instant::now();
        // `acquire_owned` consumes an `Arc<Semaphore>`; clone it so `gate`
        // stays intact for the post-acquire occupancy read below.
        match gate.semaphore.clone().acquire_owned().await {
            Ok(permit) => Some((permit, started.elapsed(), gate.held())),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "extraction gate semaphore closed; proceeding without a permit (fail-open)"
                );
                None
            }
        }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    temperature: f32,
    /// Pinning the RNG makes the extractor deterministic when paired with
    /// `temperature: 0.0`: the same input re-yields the same bullet list, so
    /// `FactId = SHA1(content)` stays stable across re-extraction runs.
    seed: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// Disables Qwen3.5 thinking mode. Qwen3.5 dropped the Qwen3-era
/// `/no_think` soft switch: thinking is toggled only via the
/// `enable_thinking` chat-template variable, which llama-server reads from
/// the nested `chat_template_kwargs` object (the top-level `enable_thinking`
/// key is silently ignored).
#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: Option<ChatResponseBody>,
}

#[derive(Deserialize)]
struct ChatResponseBody {
    content: String,
}

impl LlmExtractor for OllamaExtractor {
    async fn extract_facts(
        &self,
        response_content: &str,
        _tool_calls: &[ToolCall],
    ) -> Result<Vec<String>, ProviderError> {
        let body = ChatRequest {
            model: &self.config.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: EXTRACTION_SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user",
                    content: format!(
                        "Text:\n{response_content}\n\nFacts (one per line, starting with \"- \"):",
                    ),
                },
            ],
            stream: false,
            temperature: self.config.temperature,
            seed: self.config.seed,
            chat_template_kwargs: Some(ChatTemplateKwargs {
                enable_thinking: false,
            }),
        };

        // Acquire the extraction-slot gate (if wired) so concurrent background
        // extractions cannot pile up on a shared single-slot upstream and
        // starve chat-completion forwards. The permit is held for the whole
        // HTTP round-trip and released on drop at function return.
        let gate = self.acquire_gate().await;
        if let Some((_permit, wait, concurrent)) = &gate {
            let max = self
                .extraction_slot
                .as_ref()
                .map(|g| g.permits)
                .unwrap_or(0);
            tracing::debug!(
                target: "smos.extraction.gate",
                gate_wait_ms = wait.as_millis() as u64,
                concurrent,
                max,
                "extraction gate permit acquired"
            );
        }

        let response = match self.client.post(self.chat_url()).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                if e.is_timeout() {
                    return Err(ProviderError::Timeout(std::time::Duration::from_secs(
                        self.config.timeout_seconds,
                    )));
                }
                // Connection refused / DNS / TLS — the model is unreachable;
                // the use case treats this as a graceful skip.
                return Err(ProviderError::Unavailable(e.to_string()));
            }
        };
        let status = response.status();
        if !status.is_success() {
            return Err(ProviderError::RequestFailed(format!(
                "llama-server /v1/chat/completions returned {}",
                status
            )));
        }
        let parsed: ChatResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::InvalidResponse(format!("decode chat body: {e}")))?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message)
            .map(|m| m.content)
            .unwrap_or_default();
        Ok(parse_bullet_facts(&content))
    }
}

/// Parse a model reply into clean fact strings.
///
/// Accepts `- `, `* `, and `N. ` bullets; strips the marker; drops empty
/// lines, prompt echoes (the model occasionally restates the instructions),
/// and over-long paragraphs that are clearly not standalone facts.
pub fn parse_bullet_facts(raw_response: &str) -> Vec<String> {
    raw_response
        .lines()
        .filter_map(strip_bullet_marker)
        .filter(|fact| {
            // Single char-count pass: keep standalone assertions (>5 chars),
            // drop prompt echoes and over-long paragraphs (>500 chars).
            let len = fact.chars().count();
            len > 5 && len <= 500
        })
        .filter(|fact| !is_prompt_echo(fact))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Strip a leading bullet marker (`- `, `* `, or `N. `) from a line. Returns
/// `None` for lines that are not bullets (headers, prose, blank lines).
fn strip_bullet_marker(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("-\t"))
    {
        return Some(rest.to_string());
    }
    // Numbered bullet: digits followed by ". " or ".\t".
    let bytes = trimmed.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    let numbered = idx > 0
        && idx + 1 < bytes.len()
        && bytes[idx] == b'.'
        && (bytes[idx + 1] == b' ' || bytes[idx + 1] == b'\t');
    if numbered {
        return Some(trimmed[idx + 2..].to_string());
    }
    None
}

/// Detect lines the model emitted by echoing the prompt back (not facts).
/// Match is case-insensitive on the leading phrase.
fn is_prompt_echo(fact: &str) -> bool {
    let lower = fact.to_lowercase();
    PROMPT_ECHO_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

const PROMPT_ECHO_PREFIXES: &[&str] = &[
    "thinking process",
    "analyze the",
    "task:",
    "do not extract",
    "now extract",
    "quality over quantity",
    "each fact is",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(url: &str) -> Arc<LlmExtractionConfig> {
        Arc::new(LlmExtractionConfig {
            url: url.into(),
            model: "qwen3.5-2b".into(),
            timeout_seconds: 2,
            ..LlmExtractionConfig::default()
        })
    }

    #[test]
    fn chat_url_strips_trailing_slash_and_appends_path() {
        let ext = OllamaExtractor::new(cfg("http://llama:28082/")).expect("build");
        assert_eq!(ext.chat_url(), "http://llama:28082/v1/chat/completions");
    }

    #[test]
    fn chat_url_for_plain_base() {
        let ext = OllamaExtractor::new(cfg("http://llama:28082")).expect("build");
        assert_eq!(ext.chat_url(), "http://llama:28082/v1/chat/completions");
    }

    #[test]
    fn chat_request_disables_thinking_via_nested_chat_template_kwargs() {
        let body = ChatRequest {
            model: "qwen3.5-2b",
            messages: vec![ChatMessage {
                role: "user",
                content: "hi".to_string(),
            }],
            stream: false,
            temperature: 0.0,
            seed: 42,
            chat_template_kwargs: Some(ChatTemplateKwargs {
                enable_thinking: false,
            }),
        };
        let json = serde_json::to_value(&body).expect("serialize ChatRequest");
        assert_eq!(
            json["chat_template_kwargs"]["enable_thinking"], false,
            "thinking must be disabled inside the nested chat_template_kwargs \
             object; llama-server ignores a top-level enable_thinking key"
        );
    }

    #[test]
    fn chat_request_omits_chat_template_kwargs_when_none() {
        let body = ChatRequest {
            model: "qwen3.5-2b",
            messages: vec![],
            stream: false,
            temperature: 0.0,
            seed: 42,
            chat_template_kwargs: None,
        };
        let json = serde_json::to_value(&body).expect("serialize ChatRequest");
        assert!(
            json.get("chat_template_kwargs").is_none(),
            "chat_template_kwargs must be absent when None so requests for \
             models that do not read it stay byte-identical"
        );
    }

    #[test]
    fn parse_dash_bullets() {
        let out = parse_bullet_facts("- fact one\n- fact two");
        assert_eq!(out, vec!["fact one".to_string(), "fact two".to_string()]);
    }

    #[test]
    fn parse_asterisk_bullets() {
        let out = parse_bullet_facts("* fact one\n* fact two");
        assert_eq!(out, vec!["fact one".to_string(), "fact two".to_string()]);
    }

    #[test]
    fn parse_numbered_bullets() {
        let out = parse_bullet_facts("1. first fact\n2. second fact");
        assert_eq!(
            out,
            vec!["first fact".to_string(), "second fact".to_string()]
        );
    }

    #[test]
    fn parse_ignores_non_bullet_lines() {
        let out = parse_bullet_facts("Facts:\n- real fact\nsome prose\n- another");
        assert_eq!(out, vec!["real fact".to_string(), "another".to_string()]);
    }

    #[test]
    fn parse_filters_prompt_echoes() {
        let raw = "\
Thinking Process: analyze the input\n\
Task: extract facts\n\
- Do not extract trivial actions\n\
- real knowledge fact here";
        let out = parse_bullet_facts(raw);
        assert_eq!(out, vec!["real knowledge fact here".to_string()]);
    }

    #[test]
    fn parse_drops_too_short_facts() {
        let out = parse_bullet_facts("- ok\n- a real fact");
        assert_eq!(out, vec!["a real fact".to_string()]);
    }

    #[test]
    fn parse_drops_overlong_paragraphs() {
        let long = format!("- {}", "x".repeat(600));
        let out = parse_bullet_facts(&long);
        assert!(out.is_empty(), "500+ char paragraph is not a fact");
    }

    #[test]
    fn parse_empty_response_yields_empty() {
        assert!(parse_bullet_facts("").is_empty());
        assert!(parse_bullet_facts("no bullets here at all").is_empty());
    }
}
