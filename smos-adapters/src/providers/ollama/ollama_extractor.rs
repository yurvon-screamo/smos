//! `OllamaExtractor` — `LlmExtractor` against the Ollama `/api/chat` endpoint
//! with a Qwen3.5-class model (POC parity with `smos/extract.py`).
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

use reqwest::Client;
use serde::{Deserialize, Serialize};
use smos_application::errors::ProviderError;
use smos_application::ports::LlmExtractor;
use smos_domain::chat::ToolCall;

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

/// Ollama-backed fact extractor (Qwen3.5-2B by default).
#[derive(Clone)]
pub struct OllamaExtractor {
    client: Client,
    config: Arc<LlmExtractionConfig>,
}

impl OllamaExtractor {
    /// Build the adapter with a pooled HTTP client sized to the config timeout.
    /// Construction does NOT contact the server.
    pub fn new(config: Arc<LlmExtractionConfig>) -> Result<Self, ProviderError> {
        let client = build_client(config.timeout_seconds)?;
        Ok(Self { client, config })
    }

    fn chat_url(&self) -> String {
        format!("{}/api/chat", self.config.url.trim_end_matches('/'))
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    options: ChatOptions,
    // Qwen3-class models honour `think: false` to suppress reasoning output;
    // servers that ignore the field are unaffected.
    think: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: String,
}

#[derive(Serialize)]
struct ChatOptions {
    temperature: f32,
    /// Pinning the RNG makes the extractor deterministic when paired with
    /// `temperature: 0.0`: the same input re-yields the same bullet list, so
    /// `FactId = SHA1(content)` stays stable across re-extraction runs.
    seed: u32,
}

#[derive(Deserialize)]
struct ChatResponse {
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
            options: ChatOptions {
                temperature: self.config.temperature,
                seed: self.config.seed,
            },
            think: false,
        };

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
                "ollama /api/chat returned {}",
                status
            )));
        }
        let parsed: ChatResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::InvalidResponse(format!("decode chat body: {e}")))?;
        let content = parsed.message.map(|m| m.content).unwrap_or_default();
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
            model: "qwen3.5:2b".into(),
            timeout_seconds: 2,
            ..LlmExtractionConfig::default()
        })
    }

    #[test]
    fn chat_url_strips_trailing_slash_and_appends_path() {
        let ext = OllamaExtractor::new(cfg("http://ollama:11434/")).expect("build");
        assert_eq!(ext.chat_url(), "http://ollama:11434/api/chat");
    }

    #[test]
    fn chat_url_for_plain_base() {
        let ext = OllamaExtractor::new(cfg("http://ollama:11434")).expect("build");
        assert_eq!(ext.chat_url(), "http://ollama:11434/api/chat");
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
