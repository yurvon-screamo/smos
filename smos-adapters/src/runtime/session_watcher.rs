//! `SessionWatcher` — background sweeper that triggers `FinalizeSession` for
//! expired / overflowed sessions (§5 session-end triggers).
//!
//! Mirrors the POC `_session_sweeper` (`smos/server.py`): a single long-lived
//! tokio task that wakes on a fixed cadence, finds sessions whose
//! `last_active` is older than the inactivity timeout, and runs the full NLI
//! finalize pipeline against their pending backlog. A second scan in the same
//! cycle picks up sessions whose pending backlog crossed the overflow
//! threshold mid-conversation — so a 30-minute idle is not the only trigger.
//!
//! # Why errors are swallowed
//!
//! The watcher is the only mechanism that retires pending facts without an
//! operator pressing `smos finalize`. If a transient error (SurrealDB hiccup,
//! NLI backend restart) terminated the loop, every subsequent session would
//! pin its pending facts forever — the proxy would silently degrade into a
//! write-only memory. So every cycle error is logged at `WARN`/`ERROR` and
//! the loop continues; only the explicit shutdown channel terminates the
//! task.
//!
//! # Concurrency seam
//!
//! Per the POC `_inflight_session_end` set: the same `SessionId` must never
//! be finalized twice in parallel — `FinalizeSession` snapshots
//! `source_sessions`-owned pending ids at entry, so two concurrent runs
//! would each load the same pending pool, classify every pair twice, and
//! race on the merge saves. The `inflight` guard is the single concurrency
//! seam: marked before the call, removed after (via an RAII guard whose
//! `Drop` runs on normal return, error propagation, AND panic-unwind),
//! checked under a `std::sync::Mutex`. Different session ids may still be
//! processed concurrently if a future refactor parallelises the per-cycle
//! scan (the spec's `run_cycle` awaits sequentially, mirroring the POC
//! `_process_overflow_sessions`).
//!
//! # Why `into_loop` instead of `spawn`
//!
//! The `smos-application` port traits use native `async fn in trait` without
//! a `Send` bound on the implicit `Future` (see `ports/mod.rs`). That choice
//! keeps the ports runtime-agnostic but means a generic `spawn` method on
//! the watcher cannot satisfy `tokio::spawn`'s `F: Future + Send + 'static`
//! bound for abstract `FR`/`SR`/`NC`. The watcher therefore returns its loop
//! as `impl Future<Output = ()>` from [`into_loop`]; the caller (production
//! `smos serve` or an integration test) invokes `tokio::spawn` at a
//! concrete-type call site where the `Send` proof is trivially discharged
//! (`SurrealStore` + `NativeNliClassifier` / mock classifier all return
//! `Send` futures).

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use smos_application::errors::UseCaseError;
use smos_application::ports::{FactRepository, NliClassifier, SessionRepository};
use smos_application::use_cases::FinalizeSession;
use smos_domain::config::{ConfidenceConfig, MergeConfig, NliConfig};
use smos_domain::{MemoryKey, SessionId};
use tokio::sync::mpsc::Receiver;

use crate::config::{ServerConfig, SessionConfig};

/// Background sweeper that triggers `FinalizeSession` for expired or
/// overflowed sessions on a fixed cadence.
///
/// Cheap to construct; call [`into_loop`](Self::into_loop) to obtain the
/// loop future and spawn it on a runtime (production `smos serve` uses
/// `tokio::spawn` so the watcher runs alongside the axum server; tests do
/// the same against concrete mock-backed types).
pub struct SessionWatcher<FR, SR, NC> {
    facts: FR,
    sessions: SR,
    classifier: NC,
    confidence_cfg: Arc<ConfidenceConfig>,
    nli_cfg: Arc<NliConfig>,
    merge_cfg: Arc<MergeConfig>,
    session_cfg: Arc<SessionConfig>,
    /// TOTAL budget for the `drain_all` pass on shutdown. Same field + same
    /// semantics as `ExtractionSupervisor::drain` (§12): a single
    /// operator-facing knob bounds BOTH the extraction drain AND the session
    /// drain, so a deploy can configure `terminationGracePeriodSeconds`
    /// (K8s) / `TimeoutStopSec` (systemd) to one value that covers the
    /// whole shutdown sequence. With N sessions and budget B, total drain
    /// time ≤ B (NOT N×B) — a wedged session consuming the whole budget
    /// skips the remaining sessions, whose pending facts stay pending for
    /// the next process start.
    server_cfg: Arc<ServerConfig>,
    /// Sync mutex guarding the in-flight session set.
    ///
    /// Deliberately `std::sync::Mutex` (not `tokio::sync::Mutex`): the only
    /// operations on this set are `contains` / `insert` / `remove`, all
    /// non-async and sub-microsecond. Using the std variant makes the
    /// [`InflightGuard`] RAII pattern sound: `std::sync::MutexGuard`'s
    /// `Drop` is synchronous (it just releases the lock), so a guard that
    /// owns a guard can run its own `Drop` even during a panic unwind.
    /// `tokio::sync::Mutex`'s `Drop` would need an async context to lock
    /// again, which is unavailable inside `Drop` — so a panicking finalize
    /// task would leak the session id from the set and the next cycle
    /// would skip it forever. The std mutex closes that hole.
    inflight: Arc<StdMutex<HashSet<SessionId>>>,
}

impl<FR, SR, NC> SessionWatcher<FR, SR, NC>
where
    FR: FactRepository + Send + Sync + 'static,
    SR: SessionRepository + Send + Sync + 'static,
    NC: NliClassifier + Send + Sync + 'static,
    Self: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        facts: FR,
        sessions: SR,
        classifier: NC,
        confidence_cfg: Arc<ConfidenceConfig>,
        nli_cfg: Arc<NliConfig>,
        merge_cfg: Arc<MergeConfig>,
        session_cfg: Arc<SessionConfig>,
        server_cfg: Arc<ServerConfig>,
    ) -> Self {
        Self {
            facts,
            sessions,
            classifier,
            confidence_cfg,
            nli_cfg,
            merge_cfg,
            session_cfg,
            server_cfg,
            inflight: Arc::new(StdMutex::new(HashSet::new())),
        }
    }

    /// Build the background loop future. The caller is responsible for
    /// spawning it on a runtime — production `smos serve` and the
    /// integration tests both use `tokio::spawn` against concrete types so
    /// the resulting future's `Send` bound discharges at the call site.
    ///
    /// The loop sleeps for `scan_interval_seconds` between cycles (POC
    /// `_SESSION_CHECK_INTERVAL_SECONDS = 60`). The shutdown channel is
    /// checked via `tokio::select!` so a shutdown request that arrives
    /// mid-sleep is observed as soon as the next select! poll fires (no
    /// full scan interval of dead-time). A shutdown that arrives mid-cycle
    /// (while `run_cycle` is awaited) is deferred until the cycle completes
    /// — bounded by the per-cycle latency, which scales with the pending
    /// backlog of every session scanned. Operators should set
    /// `shutdown_extraction_grace_seconds` to bound the per-session drain
    /// inside the subsequent `drain_all` pass.
    pub fn into_loop(self, mut shutdown_rx: Receiver<()>) -> impl Future<Output = ()> {
        let scan_interval = Duration::from_secs(self.session_cfg.scan_interval_seconds);
        let session_timeout = Duration::from_secs(self.session_cfg.timeout_seconds);
        let overflow_threshold = self.session_cfg.pending_overflow_threshold;

        tracing::info!(
            scan_interval_secs = scan_interval.as_secs(),
            timeout_secs = session_timeout.as_secs(),
            overflow_threshold,
            "session watcher started"
        );

        async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(scan_interval) => {
                        if let Err(e) = self
                            .run_cycle(session_timeout, overflow_threshold)
                            .await
                        {
                            // Graceful loop: a cycle failure must NOT
                            // terminate the watcher. Transient store errors
                            // would otherwise pin every pending fact forever.
                            tracing::error!(
                                error = %e,
                                "watcher cycle failed (non-fatal, continuing)"
                            );
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        tracing::info!("session watcher received shutdown signal");
                        if let Err(e) = self.drain_all().await {
                            tracing::error!(
                                error = %e,
                                "watcher drain-all failed (best-effort)"
                            );
                        }
                        tracing::info!("session watcher stopped");
                        break;
                    }
                }
            }
        }
    }

    /// One scan cycle: drain expired sessions, then drain overflow sessions.
    /// Both halves are sequential (POC `_process_overflow_sessions` inlined
    /// into the cycle) so the `inflight` guard fully serialises per-session.
    async fn run_cycle(
        &self,
        timeout: Duration,
        overflow_threshold: usize,
    ) -> Result<(), UseCaseError> {
        // §5 trigger 1: 30-minute inactivity. Each expired session carries
        // its own `memory_key` (the watcher reads it off the SessionState
        // and forwards it to FinalizeSession — see `try_finalize` docs for
        // why ownership no longer comes from SessionState.pending_facts()).
        let expired = self.sessions.collect_expired(timeout).await?;
        for (id, state) in &expired {
            self.try_finalize(id, state.memory_key()).await?;
        }

        // §5 trigger 2: pending backlog overflow. `snapshot_all` is also the
        // cheap way to read `pending_facts().len()` for every session; the
        // `collect_expired` half above does not surface the count. Note
        // that overflow detection still reads SessionState.pending_facts()
        // — this is purely the SCHEDULING signal, not the ownership signal.
        // A session that holds ghost ids (operator-wiped facts) will
        // trigger finalize, which now correctly drains nothing (the facts
        // are gone) and clears the ghost ids from the bookkeeping.
        let all = self.sessions.snapshot_all().await?;
        for (id, state) in &all {
            if state.pending_facts().len() >= overflow_threshold {
                self.try_finalize(id, state.memory_key()).await?;
            }
        }

        Ok(())
    }

    /// Run `FinalizeSession` for `session_id` (scoped to `memory_key`) if
    /// not already in flight.
    ///
    /// The `memory_key` is sourced from the triggering `SessionState` —
    /// the watcher already has the row in hand (`collect_expired` /
    /// `snapshot_all`), so it forwards the namespace rather than
    /// re-discovering it. `FinalizeSession::execute` no longer reads
    /// `SessionState.pending_facts()` for ownership; it derives ownership
    /// from `Fact.source_sessions`. The SessionState here is solely the
    /// scheduling + namespace signal.
    ///
    /// The `inflight` set is checked + mutated atomically under the mutex,
    /// then the (potentially long) finalize call runs OUTSIDE the lock so
    /// different session ids can finalise in parallel if a future refactor
    /// parallelises the per-cycle scan. Per-session serialisation is the
    /// invariant; per-cycle serialisation is an artifact of the current
    /// awaited-loop design, not a contract.
    ///
    /// The session id is removed from `inflight` via an RAII
    /// [`InflightGuard`]: the guard's `Drop` runs on the normal path, on
    /// error returns, AND on panic-unwind, so a panicking finalize task
    /// can never pin its id in the set. (This is the reason `inflight`
    /// uses `std::sync::Mutex` — its guard's `Drop` is sync and
    /// panic-safe; `tokio::sync::Mutex` would need an async context that
    /// is unavailable inside `Drop`.)
    ///
    /// Finalize errors are swallowed: a transient NLI backend / store failure
    /// must not skip subsequent sessions in the same cycle. The cycle-level
    /// error path (storage unavailable, etc.) is the only `Err` returned.
    async fn try_finalize(
        &self,
        session_id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Result<(), UseCaseError> {
        // Mark in-flight under the lock; bail if another task already owns
        // this session. `lock().unwrap_or_else(into_inner)` recovers from a
        // poisoned mutex — std mutexes poison on panic, but our `Drop` guard
        // runs before the poisoning propagates, so the data is still
        // consistent. A panic in one finalize task must not wedge every
        // subsequent cycle.
        {
            let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
            if inflight.contains(session_id) {
                tracing::debug!(session = %session_id, "finalize already in-flight; skipping");
                return Ok(());
            }
            inflight.insert(session_id.clone());
        }

        // RAII guard: removes `session_id` on drop — covers normal return,
        // early `?`-return, and panic-unwind. Held across the (long)
        // finalize call so the lock is NOT held while finalize runs.
        let _guard = InflightGuard {
            inflight: Arc::clone(&self.inflight),
            id: Some(session_id.clone()),
        };

        let result = {
            let finalize = FinalizeSession {
                facts: &self.facts,
                sessions: &self.sessions,
                classifier: &self.classifier,
                confidence_cfg: &self.confidence_cfg,
                nli_cfg: &self.nli_cfg,
                merge_cfg: &self.merge_cfg,
            };
            finalize.execute(session_id, memory_key).await
        };

        match result {
            Ok(stats) => {
                if stats.processed > 0 {
                    tracing::info!(
                        session = %session_id,
                        memory_key = %memory_key,
                        processed = stats.processed,
                        finalized = stats.finalized,
                        merged = stats.merged,
                        conflicts = stats.conflicts,
                        "watcher finalized session"
                    );
                }
                Ok(())
            }
            Err(e) => {
                tracing::warn!(
                    session = %session_id,
                    memory_key = %memory_key,
                    error = %e,
                    "finalize failed (non-fatal, swallowed)"
                );
                Ok(())
            }
        }
    }

    /// Best-effort drain of every still-tracked session on shutdown (§12
    /// graceful shutdown). Bypasses the expiry check so a session whose
    /// `last_active` is recent still gets its pending backlog resolved
    /// before the process exits — otherwise the in-memory session tracker
    /// would drop with the pending facts never reaching `Accepted` /
    /// `Rejected`.
    ///
    /// The WHOLE pass is bounded by `server_cfg.shutdown_extraction_grace_seconds`
    /// — a TOTAL budget shared with the extraction-supervisor drain so a
    /// deploy can configure `terminationGracePeriodSeconds` (K8s) /
    /// `TimeoutStopSec` (systemd) to a single value that covers the full
    /// §12 sequence. With N sessions and budget B, total drain time ≤ B
    /// (NOT N×B): a wedged NLI backend / pathological pending backlog
    /// consumes the remaining budget, the loop breaks, and the
    /// unprocessed sessions' pending facts stay pending for the next
    /// process start (graceful degradation, not data loss).
    async fn drain_all(&self) -> Result<(), UseCaseError> {
        let total_budget = Duration::from_secs(self.server_cfg.shutdown_extraction_grace_seconds);
        let deadline = tokio::time::Instant::now() + total_budget;
        tracing::info!(
            total_budget_secs = total_budget.as_secs(),
            "draining all sessions on shutdown"
        );
        let all = self.sessions.snapshot_all().await?;
        let total = all.len();
        let mut processed = 0usize;
        let mut timed_out = 0usize;

        for (id, state) in all {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                tracing::warn!(
                    budget_secs = total_budget.as_secs(),
                    remaining = total - processed,
                    "drain budget exhausted; skipping remaining sessions \
                     (their pending facts stay pending for the next start)"
                );
                break;
            }
            let remaining = deadline - now;
            // The drain forwards the SessionState's memory_key so the
            // finalize call is scoped to the right namespace even when
            // the underlying ownership (fact.source_sessions) does not
            // depend on SessionState anymore.
            let memory_key = state.memory_key().clone();
            let finalize = FinalizeSession {
                facts: &self.facts,
                sessions: &self.sessions,
                classifier: &self.classifier,
                confidence_cfg: &self.confidence_cfg,
                nli_cfg: &self.nli_cfg,
                merge_cfg: &self.merge_cfg,
            };
            // `tokio::time::timeout` returns `Result<inner::Result, Elapsed>`;
            // the outer `Err` is the timeout firing (this session ate the
            // rest of the budget — the loop breaks on the next iteration).
            match tokio::time::timeout(remaining, finalize.execute(&id, &memory_key)).await {
                Ok(Ok(stats)) => {
                    processed += 1;
                    tracing::info!(
                        session = %id,
                        memory_key = %memory_key,
                        processed_facts = stats.processed,
                        finalized = stats.finalized,
                        "drain progress {}/{}",
                        processed,
                        total,
                    );
                }
                Ok(Err(e)) => {
                    processed += 1;
                    tracing::warn!(
                        session = %id,
                        memory_key = %memory_key,
                        error = %e,
                        "drain finalize failed (continuing)"
                    );
                }
                Err(_elapsed) => {
                    timed_out += 1;
                    tracing::warn!(
                        session = %id,
                        memory_key = %memory_key,
                        budget_secs = total_budget.as_secs(),
                        "drain finalize exceeded remaining budget; stopping drain \
                         (remaining sessions stay pending for the next start)"
                    );
                    break;
                }
            }
        }

        tracing::info!(
            total,
            processed,
            timed_out,
            "session watcher drain complete"
        );
        Ok(())
    }
}

/// RAII guard that removes a session id from the watcher's `inflight` set
/// on drop.
///
/// Created AFTER the id has been inserted under the lock; dropped when the
/// surrounding scope exits — whether via normal return, `?` propagation, or
/// panic-unwind. The `Option<SessionId>` field lets `Drop` distinguish
/// "still owned" from "already taken" so a future caller can hand off
/// ownership (e.g., move the id into a follow-up task) without double
/// removing.
///
/// Uses `std::sync::Mutex` because the only operations are `lock` +
/// `HashSet::remove`, both synchronous and sub-microsecond. `tokio::sync`
/// would forbid the lock acquisition from inside `Drop` (no async context
/// available during unwinding), so a panicking finalize task would leak its
/// id into the set forever.
struct InflightGuard {
    inflight: Arc<StdMutex<HashSet<SessionId>>>,
    id: Option<SessionId>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            // `unwrap_or_else(into_inner)` recovers from a poisoned mutex:
            // a prior panic would have left the lock poisoned, but the
            // underlying HashSet is still consistent (the panicking task's
            // own guard ran during unwind before the poisoning propagated
            // to us). Leaking the id here would re-introduce the bug this
            // guard exists to fix.
            let mut inflight = self.inflight.lock().unwrap_or_else(|e| e.into_inner());
            inflight.remove(&id);
        }
    }
}
