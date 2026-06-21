//! `ImportOpencodeSession` — bulk import of an opencode transcript (Slice-8).
//!
//! Parses already-flattened assistant turns (the adapter layer's
//! [`AssistantTurn`] is produced by `smos_adapters::opencode::transcript`) and
//! re-runs the SAME extraction pipeline the live proxy runs after each chat
//! completion. Concretely: every turn is fed to
//! [`ExtractFactsFromResponse`], so dedup, embedding, cross-session
//! confirmation, and the `MIN_INPUT_CHARS` floor are reused verbatim — the
//! import path is DRY with the live path.
//!
//! # Filtering
//!
//! The use case applies two pre-extraction filters that mirror the POC
//! `iter_assistant_turns`:
//!
//! 1. **Agent filter** — optional `&[String]` allow-list. Turns whose `agent`
//!    is not in the list are skipped (`turns_skipped`).
//! 2. **Min-chars floor** — turns with fewer than `min_chars` content chars AND
//!    no tool calls are skipped. Tool-call-only turns survive because the
//!    extraction pipeline renders tool calls into the input, so a turn with
//!    zero prose still carries extractable signal.
//!
//! `min_chars` is wired from the SAME const as the live extraction pipeline
//! ([`extract_facts_from_response::MIN_INPUT_CHARS`]) by the CLI binary, so
//! the import path and the live response path cannot drift apart. The use
//! case keeps the field as a runtime knob (not a const) so future callers
//! can override it explicitly when they have a stronger reason than "match
//! the live path".
//!
//! # Stats
//!
//! [`ImportStats`] is the wire shape surfaced by the `smos-import` binary. The
//! `facts_extracted` counter is the sum of `ExtractFactsFromResponse::execute`
//! return values — i.e. ONLY newly-created pending facts. Cross-session
//! confirmations on pre-existing facts do NOT increment the counter (they
//! update an existing fact's provenance instead), so re-importing the same
//! session is idempotent on the new-fact axis.

use std::sync::Arc;

use smos_domain::chat::ToolCall;
use smos_domain::config::{ConfidenceConfig, ExtractionConfig};
use smos_domain::{MemoryKey, SessionId};

use crate::errors::UseCaseError;
use crate::ports::{
    Clock, Delay, EmbeddingProvider, FactRepository, LlmExtractor, SessionRepository,
};
use crate::use_cases::extract_facts_from_response::ExtractFactsFromResponse;

/// One assistant turn parsed from an opencode transcript.
///
/// Pure data — no IO concerns. Produced by
/// `smos_adapters::opencode::transcript::parse_transcript` and consumed by
/// [`ImportOpencodeSession::execute`].
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantTurn {
    pub message_id: String,
    pub agent: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Aggregate outcome counters for one import run.
///
/// Surfaced to operators by the `smos-import` CLI. `facts_extracted` is the
/// number of NEWLY-stored pending facts (cross-session confirmations on
/// pre-existing facts do NOT count — see the module docs for the idempotency
/// contract).
#[derive(Debug, Clone, Default)]
pub struct ImportStats {
    pub session_id: String,
    pub turns_processed: usize,
    pub turns_skipped: usize,
    pub facts_extracted: usize,
}

/// Import an opencode transcript by re-running the live extraction pipeline.
///
/// Owns the same six port dependencies `ExtractFactsFromResponse` needs
/// (`facts`, `sessions`, `embedder`, `extractor`, `clock`, `delay`) plus the
/// configuration knobs the per-turn extraction relies on. The concrete
/// `TokioDelay` adapter is wired by the CLI binary; unit tests inject a
/// no-op delay so the retry backoff is instantaneous.
pub struct ImportOpencodeSession<FR, SR, EP, LE, C, D> {
    pub facts: FR,
    pub sessions: SR,
    pub embedder: EP,
    pub extractor: LE,
    pub clock: C,
    pub delay: D,
    pub confidence_cfg: Arc<ConfidenceConfig>,
    /// Semantic-dedup safety net, threaded into the per-turn
    /// [`ExtractFactsFromResponse`] bundle so the import path and the live
    /// response path share one source of truth.
    pub extraction_cfg: Arc<ExtractionConfig>,
    pub enable_response_extraction: bool,
    /// Pre-extraction content floor. Turns below this length AND without tool
    /// calls are skipped. Wired from
    /// [`extract_facts_from_response::MIN_INPUT_CHARS`] by the CLI binary so
    /// the import path and the live response path share one source of truth.
    pub min_chars: usize,
}

impl<FR, SR, EP, LE, C, D> ImportOpencodeSession<FR, SR, EP, LE, C, D>
where
    FR: FactRepository,
    SR: SessionRepository,
    EP: EmbeddingProvider,
    LE: LlmExtractor,
    C: Clock,
    D: Delay,
{
    /// Import `turns` under `(memory_key, session_id)`.
    ///
    /// Reuses [`ExtractFactsFromResponse`] per turn so the extraction contract
    /// is identical to the live response pipeline. Returns aggregate stats;
    /// never raises on a per-turn extraction failure (the extractor's retry
    /// loop already swallows transient failures per §12 fail-open).
    pub async fn execute(
        &self,
        turns: Vec<AssistantTurn>,
        memory_key: &MemoryKey,
        session_id: &SessionId,
        agent_filter: Option<&[String]>,
    ) -> Result<ImportStats, UseCaseError> {
        let mut stats = ImportStats {
            session_id: session_id.as_str().to_string(),
            ..Default::default()
        };

        // Ensure the session row exists so `add_pending` registrations land on
        // a real row. The session also serves as the cross-session
        // confirmation key inside `ExtractFactsFromResponse::persist_facts`.
        self.sessions.get_or_create(session_id, memory_key).await?;

        for turn in &turns {
            if self.should_skip(turn, agent_filter) {
                stats.turns_skipped += 1;
                continue;
            }

            stats.turns_processed += 1;
            let new_count = self.extract_turn(turn, memory_key, session_id).await?;
            stats.facts_extracted += new_count;
        }

        tracing::info!(
            session = %session_id,
            memory_key = %memory_key,
            processed = stats.turns_processed,
            skipped = stats.turns_skipped,
            new_facts = stats.facts_extracted,
            "import complete"
        );
        Ok(stats)
    }

    /// Apply the agent + min-chars filters. Returns `true` when the turn must
    /// be skipped, `false` when it should be processed.
    fn should_skip(&self, turn: &AssistantTurn, agent_filter: Option<&[String]>) -> bool {
        if let Some(filter) = agent_filter
            && !filter.iter().any(|a| a == &turn.agent)
        {
            return true;
        }
        let too_short = turn.content.chars().count() < self.min_chars;
        too_short && turn.tool_calls.is_empty()
    }

    /// Delegate one turn to `ExtractFactsFromResponse` (DRY with the live
    /// response path). The borrow bundle is rebuilt per turn so the use case
    /// does not hold references across awaits between turns.
    async fn extract_turn(
        &self,
        turn: &AssistantTurn,
        memory_key: &MemoryKey,
        session_id: &SessionId,
    ) -> Result<usize, UseCaseError> {
        let extractor = ExtractFactsFromResponse {
            facts: &self.facts,
            sessions: &self.sessions,
            embedder: &self.embedder,
            extractor: &self.extractor,
            clock: &self.clock,
            delay: &self.delay,
            confidence_cfg: &self.confidence_cfg,
            extraction_cfg: &self.extraction_cfg,
            enable_response_extraction: self.enable_response_extraction,
        };
        extractor
            .execute(&turn.content, &turn.tool_calls, memory_key, session_id)
            .await
    }
}

#[cfg(test)]
mod tests {
    //! Import use case unit tests.
    //!
    //! Classicist style: in-memory repos + scripted providers. The full
    //! pipeline (SurrealStore + extraction) is exercised by the
    //! `tests/e2e_import.rs` integration suite.

    use super::*;
    use crate::types::SearchHit;
    use smos_domain::{Fact, FactId, Heat, SessionState, Timestamp};
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;

    // ---- Fakes mirroring the `extract_facts_from_response` test kit ----

    #[derive(Clone)]
    struct FixedClock(Timestamp);
    impl Clock for FixedClock {
        fn now(&self) -> Timestamp {
            self.0
        }
    }

    #[derive(Clone, Copy)]
    struct NoOpDelay;
    impl Delay for NoOpDelay {
        async fn delay(&self, _duration: Duration) {}
    }

    struct ScriptedExtractor {
        results: Mutex<Vec<Vec<String>>>,
    }
    impl ScriptedExtractor {
        fn new(results: Vec<Vec<String>>) -> Self {
            Self {
                results: Mutex::new(results),
            }
        }
    }
    impl LlmExtractor for ScriptedExtractor {
        async fn extract_facts(
            &self,
            _content: &str,
            _tool_calls: &[ToolCall],
        ) -> Result<Vec<String>, crate::errors::ProviderError> {
            let mut guard = self.results.lock().unwrap();
            if guard.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(guard.remove(0))
            }
        }
    }

    struct ConstantEmbedder(Vec<f32>);
    impl EmbeddingProvider for ConstantEmbedder {
        async fn embed(
            &self,
            _text: &str,
        ) -> Result<Option<Vec<f32>>, crate::errors::ProviderError> {
            Ok(Some(self.0.clone()))
        }
    }

    #[derive(Default, Clone)]
    struct InMemoryFacts {
        store: std::sync::Arc<Mutex<HashMap<String, Fact>>>,
    }
    impl FactRepository for InMemoryFacts {
        async fn save(&self, fact: &Fact) -> Result<(), crate::errors::RepoError> {
            self.store
                .lock()
                .unwrap()
                .insert(fact.id().as_str().to_string(), fact.clone());
            Ok(())
        }
        async fn get(
            &self,
            id: &FactId,
            _mk: &MemoryKey,
        ) -> Result<Option<Fact>, crate::errors::RepoError> {
            Ok(self.store.lock().unwrap().get(id.as_str()).cloned())
        }
        async fn list_accepted(
            &self,
            _mk: &MemoryKey,
        ) -> Result<Vec<Fact>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn list_pending(
            &self,
            _mk: &MemoryKey,
        ) -> Result<Vec<Fact>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn list_memory_keys_for_session(
            &self,
            _session_id: &SessionId,
        ) -> Result<Vec<MemoryKey>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn search_similar(
            &self,
            _e: Vec<f32>,
            _mk: &MemoryKey,
            _l: usize,
        ) -> Result<Vec<SearchHit>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn update_heat_batch(
            &self,
            _ids: &[FactId],
            _mk: &MemoryKey,
            _h: Heat,
            _t: Timestamp,
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
    }

    #[derive(Default, Clone)]
    struct RecordingSessions {
        created: std::sync::Arc<Mutex<bool>>,
    }
    impl SessionRepository for RecordingSessions {
        async fn get_or_create(
            &self,
            id: &SessionId,
            _m: &MemoryKey,
        ) -> Result<SessionState, crate::errors::RepoError> {
            *self.created.lock().unwrap() = true;
            Ok(SessionState::new(
                id.clone(),
                MemoryKey::from_raw("proj").unwrap(),
                Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ))
        }
        async fn add_pending(
            &self,
            _i: &SessionId,
            _ids: &[FactId],
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
        async fn collect_expired(
            &self,
            _t: Duration,
        ) -> Result<Vec<(SessionId, SessionState)>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn snapshot_all(
            &self,
        ) -> Result<Vec<(SessionId, SessionState)>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn remove_pending_owned(
            &self,
            _i: &SessionId,
            _o: &[FactId],
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
        async fn clear_session(&self, _i: &SessionId) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
        async fn dedup_and_mark(
            &self,
            _i: &SessionId,
            _m: &MemoryKey,
            _c: &[FactId],
        ) -> Result<Vec<FactId>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn save(
            &self,
            _i: &SessionId,
            _s: &SessionState,
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
    }

    fn mk() -> MemoryKey {
        MemoryKey::from_raw("proj").unwrap()
    }
    fn sid(tag: u8) -> SessionId {
        SessionId::from_raw(&format!("sess_{:012x}", tag as u64)).unwrap()
    }

    struct Fix {
        facts: InMemoryFacts,
        sessions: RecordingSessions,
        embedder: ConstantEmbedder,
        clock: FixedClock,
        cfg: ConfidenceConfig,
        extraction_cfg: ExtractionConfig,
    }
    impl Fix {
        fn new() -> Self {
            Self {
                facts: InMemoryFacts::default(),
                sessions: RecordingSessions::default(),
                embedder: ConstantEmbedder(vec![0.1, 0.2, 0.3]),
                clock: FixedClock(Timestamp::from_unix_secs(1_700_000_000).unwrap()),
                cfg: ConfidenceConfig::default(),
                extraction_cfg: ExtractionConfig::default(),
            }
        }
        fn build(
            &self,
            extractor: ScriptedExtractor,
            min_chars: usize,
        ) -> ImportOpencodeSession<
            InMemoryFacts,
            RecordingSessions,
            ConstantEmbedder,
            ScriptedExtractor,
            FixedClock,
            NoOpDelay,
        > {
            ImportOpencodeSession {
                facts: self.facts.clone(),
                sessions: self.sessions.clone(),
                embedder: ConstantEmbedder(self.embedder.0.clone()),
                extractor,
                clock: FixedClock(self.clock.0),
                delay: NoOpDelay,
                confidence_cfg: Arc::new(self.cfg.clone()),
                extraction_cfg: Arc::new(self.extraction_cfg.clone()),
                enable_response_extraction: true,
                min_chars,
            }
        }
    }

    fn turn(agent: &str, content: &str) -> AssistantTurn {
        AssistantTurn {
            message_id: format!("msg_{agent}"),
            agent: agent.to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
        }
    }

    #[tokio::test]
    async fn execute_imports_each_turn_and_counts_new_facts() {
        let fix = Fix::new();
        let extractor = ScriptedExtractor::new(vec![
            vec!["fact one".to_string()],
            vec!["fact two".to_string()],
        ]);
        let import = fix.build(extractor, 15);

        let turns = vec![
            turn("head-of-development", "TTL=10 prevents refresh loop"),
            turn("head-of-development", "Auth uses JWT for tokens"),
        ];
        let stats = import.execute(turns, &mk(), &sid(1), None).await.unwrap();

        assert_eq!(stats.turns_processed, 2);
        assert_eq!(stats.turns_skipped, 0);
        assert_eq!(stats.facts_extracted, 2);
    }

    #[tokio::test]
    async fn execute_skips_turns_below_min_chars_without_tool_calls() {
        let fix = Fix::new();
        // Only one extraction result is scripted; the short turn must be
        // skipped so the second turn does not consume a result.
        let extractor = ScriptedExtractor::new(vec![vec!["real fact".to_string()]]);
        let import = fix.build(extractor, 15);

        let turns = vec![
            turn("a", "ok"), // 2 chars < 15 → skipped
            turn("a", "TTL=10 prevents refresh loop"),
        ];
        let stats = import.execute(turns, &mk(), &sid(1), None).await.unwrap();

        assert_eq!(stats.turns_processed, 1);
        assert_eq!(stats.turns_skipped, 1);
        assert_eq!(stats.facts_extracted, 1);
    }

    #[tokio::test]
    async fn execute_keeps_short_turn_when_it_has_tool_calls() {
        let fix = Fix::new();
        let extractor = ScriptedExtractor::new(vec![vec!["from tool".to_string()]]);
        let import = fix.build(extractor, 15);

        let mut short_with_tool = turn("a", "ok");
        short_with_tool.tool_calls.push(ToolCall {
            name: "read_file".into(),
            arguments: smos_domain::chat::ToolArguments::from_json(r#"{"path":"auth.rs"}"#),
        });
        let stats = import
            .execute(vec![short_with_tool], &mk(), &sid(1), None)
            .await
            .unwrap();

        assert_eq!(stats.turns_processed, 1);
        assert_eq!(stats.turns_skipped, 0);
        assert_eq!(stats.facts_extracted, 1);
    }

    #[tokio::test]
    async fn execute_applies_agent_filter() {
        let fix = Fix::new();
        let extractor = ScriptedExtractor::new(vec![
            vec!["hod fact".to_string()],
            vec!["hod fact 2".to_string()],
        ]);
        let import = fix.build(extractor, 15);

        let turns = vec![
            turn("head-of-development", "TTL=10 prevents refresh loop"),
            turn("dreaming", "Internal analysis content here"),
            turn("head-of-development", "Auth uses JWT for tokens"),
        ];
        let filter = vec!["head-of-development".to_string()];
        let stats = import
            .execute(turns, &mk(), &sid(1), Some(&filter))
            .await
            .unwrap();

        assert_eq!(stats.turns_processed, 2);
        assert_eq!(stats.turns_skipped, 1);
        assert_eq!(stats.facts_extracted, 2);
    }

    #[tokio::test]
    async fn execute_ensures_session_row_exists_before_first_turn() {
        let fix = Fix::new();
        let extractor = ScriptedExtractor::new(vec![]);
        let import = fix.build(extractor, 15);

        let _ = import.execute(vec![], &mk(), &sid(7), None).await.unwrap();

        assert!(
            *fix.sessions.created.lock().unwrap(),
            "get_or_create must run even for an empty turn list"
        );
    }

    #[tokio::test]
    async fn execute_with_extraction_disabled_returns_zero_facts() {
        let fix = Fix::new();
        let extractor = ScriptedExtractor::new(vec![vec!["should not be stored".to_string()]]);
        let mut import = fix.build(extractor, 15);
        import.enable_response_extraction = false;

        let stats = import
            .execute(
                vec![turn("a", "TTL=10 prevents refresh loop")],
                &mk(),
                &sid(1),
                None,
            )
            .await
            .unwrap();

        assert_eq!(stats.turns_processed, 1);
        assert_eq!(stats.facts_extracted, 0);
        assert!(fix.facts.store.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn execute_confirms_existing_fact_instead_of_counting_it_new() {
        let fix = Fix::new();
        // First import seeds the fact; second import re-observes it from a
        // different session → cross-session confirmation, no new count.
        let seeded_content = "shared fact content here";
        let first = Fact::new_pending(
            seeded_content,
            mk(),
            sid(1),
            smos_domain::Embedding::new(vec![1.0]).unwrap(),
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap();
        let fid = first.id().clone();
        fix.facts
            .store
            .lock()
            .unwrap()
            .insert(fid.as_str().to_string(), first);

        let extractor = ScriptedExtractor::new(vec![vec![seeded_content.to_string()]]);
        let import = fix.build(extractor, 15);

        let stats = import
            .execute(vec![turn("a", seeded_content)], &mk(), &sid(2), None)
            .await
            .unwrap();

        assert_eq!(stats.facts_extracted, 0, "confirmation is not a new fact");
        let confirmed = fix
            .facts
            .store
            .lock()
            .unwrap()
            .get(fid.as_str())
            .cloned()
            .expect("fact present");
        // Cross-session confirmation promotes the fact through the validation
        // gate (multi-source bonus + no-contradiction bonus clears accept
        // threshold). The exact status depends on the confidence formula; we
        // only assert provenance growth so the test is robust to formula
        // tweaks.
        assert_eq!(
            confirmed.source_sessions().distinct_count(),
            2,
            "provenance grew to two sessions"
        );
    }
}
