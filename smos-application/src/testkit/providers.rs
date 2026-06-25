//! Provider test doubles: scripted LLM extractor, constant/recording
//! embedders, and the dual-mode scripted NLI classifier.

use std::sync::{Arc, Mutex};

use smos_domain::NliResult;
use smos_domain::chat::ToolCall;

use crate::errors::ProviderError;
use crate::ports::{EmbeddingProvider, LlmExtractor, NliClassifier};

/// LLM extractor that pops pre-scripted results in FIFO order and counts
/// invocations. When the script is exhausted, subsequent calls return an empty
/// `Vec` (mirroring a provider that simply finds no facts) rather than
/// erroring, so tests that do not care about the Nth call still pass.
pub struct ScriptedExtractor {
    results: Mutex<Vec<Result<Vec<String>, ProviderError>>>,
    calls: Mutex<u32>,
}

impl ScriptedExtractor {
    pub fn new(results: Vec<Result<Vec<String>, ProviderError>>) -> Self {
        Self {
            results: Mutex::new(results),
            calls: Mutex::new(0),
        }
    }

    pub fn call_count(&self) -> u32 {
        *self.calls.lock().unwrap()
    }
}

impl LlmExtractor for ScriptedExtractor {
    async fn extract_facts(
        &self,
        _content: &str,
        _tool_calls: &[ToolCall],
    ) -> Result<Vec<String>, ProviderError> {
        *self.calls.lock().unwrap() += 1;
        let mut guard = self.results.lock().unwrap();
        if guard.is_empty() {
            Ok(Vec::new())
        } else {
            guard.remove(0)
        }
    }
}

/// Embedding provider that always returns the same vector regardless of input.
pub struct ConstantEmbedder(pub Vec<f32>);

impl EmbeddingProvider for ConstantEmbedder {
    async fn embed(&self, _text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
        Ok(Some(self.0.clone()))
    }
}

/// Embedding provider that records every `embed` call and returns a
/// deterministic content-derived vector unique to the input text. Used to
/// verify the extraction pipeline hands distinct embeddings to distinct facts
/// (so Layer 2 dedup makes the right call). `new` returns the double together
/// with the shared call-log handle so the test body can assert on it.
pub struct RecordingEmbedder {
    calls: Arc<Mutex<Vec<String>>>,
}

impl RecordingEmbedder {
    pub fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                calls: calls.clone(),
            },
            calls,
        )
    }

    fn vector_for(text: &str) -> Vec<f32> {
        // Stable, content-derived 1024-dim one-hot-ish vector: hash the text
        // into a single u64 and use it as the index of the single non-zero
        // dimension. Distinct inputs land on distinct indices, so the cosine
        // similarity across different hashes is 0.
        let hash = text
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let mut vec = vec![0.0; 1024];
        vec[(hash as usize) % 1024] = 1.0;
        vec
    }
}

impl EmbeddingProvider for RecordingEmbedder {
    async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
        self.calls.lock().unwrap().push(text.to_string());
        Ok(Some(Self::vector_for(text)))
    }
}

/// Closure type used by the matcher variant of [`ScriptedNliClassifier`].
type NliResolver = Box<dyn Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync>;

/// Scripted NLI classifier with two modes:
/// - [`ScriptedNliClassifier::new`] (FIFO): each call pops the next verdict
///   from the queue. Use when the test controls call order.
/// - [`ScriptedNliClassifier::matching`] (Match): each call dispatches to the
///   supplied closure. Use when pending iteration order is not deterministic
///   (`HashMap` order) and the test keys verdicts on the candidate text.
///
/// Both modes record every (premise, hypothesis) pair so tests can assert on
/// the exact set of pairs the use case asked about.
pub enum ScriptedNliClassifier {
    Fifo {
        verdicts: Mutex<Vec<Result<NliResult, ProviderError>>>,
        calls: Mutex<Vec<(String, String)>>,
    },
    Match {
        resolver: NliResolver,
        calls: Mutex<Vec<(String, String)>>,
    },
}

impl ScriptedNliClassifier {
    pub fn new(verdicts: Vec<Result<NliResult, ProviderError>>) -> Self {
        Self::Fifo {
            verdicts: Mutex::new(verdicts),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn matching<F>(resolver: F) -> Self
    where
        F: Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync + 'static,
    {
        Self::Match {
            resolver: Box::new(resolver),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<(String, String)> {
        match self {
            Self::Fifo { calls, .. } | Self::Match { calls, .. } => calls.lock().unwrap().clone(),
        }
    }
}

impl NliClassifier for ScriptedNliClassifier {
    async fn classify(&self, premise: &str, hypothesis: &str) -> Result<NliResult, ProviderError> {
        match self {
            Self::Fifo { verdicts, calls } => {
                calls
                    .lock()
                    .unwrap()
                    .push((premise.to_string(), hypothesis.to_string()));
                let mut queue = verdicts.lock().unwrap();
                if queue.is_empty() {
                    Err(ProviderError::Unavailable("scripted queue empty".into()))
                } else {
                    queue.remove(0)
                }
            }
            Self::Match { resolver, calls } => {
                calls
                    .lock()
                    .unwrap()
                    .push((premise.to_string(), hypothesis.to_string()));
                resolver(premise, hypothesis)
            }
        }
    }
}
