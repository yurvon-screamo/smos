//! `ReqwestUpstream` — OpenAI-compatible HTTP upstream via `reqwest`.
//!
//! Forwards a `ChatRequest` to the configured upstream URL and returns either:
//! - `ChatResponse::Streaming(bytes_stream)` when `request.is_streaming()`, or
//! - `ChatResponse::NonStreaming(json)` otherwise.
//!
//! The body is serialised from `ChatRequest` (its `#[serde(flatten)] extra`
//! keeps every OpenAI parameter intact on the wire). Auth uses a `Bearer`
//! token by default; the header name is configurable to support Azure-style
//! `api-key` headers.
//!
//! # Multi-provider pool
//!
//! [`ReqwestUpstreamPool`] wraps N [`ReqwestUpstream`] instances and routes
//! each request according to [`UpstreamStrategy`]:
//!
//! - `single` — always use `providers[0]`. The simplest mode, useful when the
//!   operator wants explicit control over which provider handles traffic.
//! - `round_robin` — atomic counter advances one slot per request. Even
//!   distribution across healthy providers without active health checks.
//! - `failover` — try providers in order; on `Err` log + retry the next. The
//!   first `Ok` wins; if every provider fails, returns
//!   [`UpstreamError::AllProvidersFailed`].
//!
//! `single` is the safe default for unknown `mode` values so a typo never
//! silently enables round-robin/failover.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::StreamExt;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use smos_application::errors::UpstreamError;
use smos_application::ports::LlmUpstream;
use smos_application::types::{ChatRequest, ChatResponse};

use crate::config::{UpstreamConfig, UpstreamProvider};

/// HTTP upstream backed by a pooled `reqwest::Client`.
#[derive(Clone)]
pub struct ReqwestUpstream {
    client: Client,
    provider: Arc<UpstreamProvider>,
}

impl fmt::Debug for ReqwestUpstream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestUpstream")
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl ReqwestUpstream {
    /// Build a new upstream with a request timeout configured from
    /// `provider.timeout_seconds`. Validates `api_key` (if non-empty) up front
    /// so a misconfigured secret with control characters fails fast at startup
    /// rather than silently producing an unauthenticated request later.
    pub fn new(provider: UpstreamProvider) -> Result<Self, UpstreamError> {
        if !provider.api_key.is_empty()
            && let Err(e) = HeaderValue::from_str(&provider.api_key)
        {
            return Err(UpstreamError::ConnectFailed(format!(
                "api_key contains invalid header bytes: {e}"
            )));
        }
        let timeout = Duration::from_secs(provider.timeout_seconds);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| UpstreamError::ConnectFailed(e.to_string()))?;
        Ok(Self {
            client,
            provider: Arc::new(provider),
        })
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if !self.provider.api_key.is_empty() {
            // Azure-style `api-key:` header carries the raw key; every other
            // header name uses the `Authorization: Bearer <key>` scheme.
            let is_api_key_header = self.provider.auth_header.eq_ignore_ascii_case("api-key");
            let value = if is_api_key_header {
                HeaderValue::from_str(&self.provider.api_key)
            } else {
                HeaderValue::from_str(&format!("Bearer {}", self.provider.api_key))
            }
            .expect("api_key validated at construction");
            let name = safe_header_name(&self.provider.auth_header);
            headers.insert(name, value);
        }
        headers
    }

    fn provider_name(&self) -> &str {
        &self.provider.name
    }

    async fn send(&self, request: &ChatRequest) -> Result<reqwest::Response, UpstreamError> {
        let body = serde_json::to_value(request)
            .map_err(|e| UpstreamError::SerializationError(e.to_string()))?;
        let response = self
            .client
            .post(&self.provider.url)
            .headers(self.build_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| map_send_error(e, self.provider.timeout_seconds))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(UpstreamError::StatusError {
                status: status.as_u16(),
                body: text,
            });
        }
        Ok(response)
    }
}

impl LlmUpstream for ReqwestUpstream {
    async fn complete(&self, request: ChatRequest) -> Result<ChatResponse, UpstreamError> {
        let is_streaming = request.is_streaming();
        let response = self.send(&request).await?;
        if is_streaming {
            let stream = response
                .bytes_stream()
                .map(|item| item.map_err(|e| UpstreamError::StreamError(e.to_string())));
            Ok(ChatResponse::Streaming(Box::new(stream)))
        } else {
            let value = response
                .json::<serde_json::Value>()
                .await
                .map_err(|e| UpstreamError::BadResponse(e.to_string()))?;
            Ok(ChatResponse::NonStreaming(value))
        }
    }
}

/// Pool of [`ReqwestUpstream`] instances backing a single [`LlmUpstream`]
/// trait surface. Routing is decided per request by [`UpstreamStrategy`].
///
/// Construction fails fast if any provider's `api_key` carries invalid header
/// bytes — partial construction would leave a pool with N-1 working
/// providers and one silently broken, which is the exact failure mode the
/// multi-provider design is supposed to surface up front.
///
/// Cheap to clone: every clone shares the same inner state (providers,
/// strategy, atomic round-robin counter) via `Arc`. The HTTP-side fan-out
/// therefore observes a single global round-robin position across all
/// router clones, which matches the operator's mental model of one
/// "logical" upstream even though axum hands each request a fresh
/// [`AppState`](crate::http::axum_server::AppState) snapshot.
#[derive(Clone, Debug)]
pub struct ReqwestUpstreamPool {
    inner: Arc<PoolInner>,
}

#[derive(Debug)]
struct PoolInner {
    providers: Vec<ReqwestUpstream>,
    strategy_mode: StrategyMode,
    counter: AtomicUsize,
}

impl ReqwestUpstreamPool {
    /// Build a pool from the configured provider list of an
    /// [`UpstreamConfig`]. The pool takes ownership of its
    /// [`ReqwestUpstream`] instances so the config struct can be dropped
    /// after wiring.
    pub fn new(config: &UpstreamConfig) -> Result<Self, UpstreamError> {
        let providers_raw = &config.providers;
        if providers_raw.is_empty() {
            return Err(UpstreamError::ConnectFailed(
                "no upstream providers configured".into(),
            ));
        }
        let mut providers = Vec::with_capacity(providers_raw.len());
        for (idx, provider) in providers_raw.iter().enumerate() {
            // Surface the offending provider's index + name in the error
            // chain so an operator with N configured providers can tell
            // which entry has the bad api_key. Without this annotation the
            // underlying `UpstreamError::ConnectFailed` only carries the
            // generic "api_key contains invalid header bytes" message.
            let provider_name = provider.name.clone();
            let upstream = ReqwestUpstream::new(provider.clone()).map_err(|e| {
                UpstreamError::ConnectFailed(format!(
                    "provider[{idx}] (name={provider_name:?}) rejected: {e}"
                ))
            })?;
            providers.push(upstream);
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                providers,
                strategy_mode: StrategyMode::from_str_logged(&config.strategy.mode),
                counter: AtomicUsize::new(0),
            }),
        })
    }

    /// Number of providers in the pool. Exposed for diagnostics / tests.
    pub fn provider_count(&self) -> usize {
        self.inner.providers.len()
    }
}

impl LlmUpstream for ReqwestUpstreamPool {
    async fn complete(&self, request: ChatRequest) -> Result<ChatResponse, UpstreamError> {
        match self.inner.strategy_mode {
            StrategyMode::Single => self.inner.providers[0].complete(request).await,
            StrategyMode::RoundRobin => {
                let idx = self
                    .inner
                    .counter
                    .fetch_add(1, Ordering::Relaxed)
                    .wrapping_rem(self.inner.providers.len());
                self.inner.providers[idx].complete(request).await
            }
            StrategyMode::Failover => {
                let mut last_err: Option<UpstreamError> = None;
                for provider in &self.inner.providers {
                    match provider.complete(request.clone()).await {
                        Ok(resp) => return Ok(resp),
                        Err(e) => {
                            tracing::warn!(
                                provider = provider.provider_name(),
                                error = %e,
                                "upstream provider failed; trying next"
                            );
                            last_err = Some(e);
                        }
                    }
                }
                Err(UpstreamError::AllProvidersFailed(
                    last_err
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "no providers attempted".into()),
                ))
            }
        }
    }
}

/// Internal enum mirroring [`UpstreamStrategy::mode`] without the string
/// parsing cost on every request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrategyMode {
    Single,
    RoundRobin,
    Failover,
}

impl StrategyMode {
    fn from_str_logged(raw: &str) -> Self {
        let parsed = Self::from_str(raw);
        if parsed == Self::Single
            && !raw.trim().eq_ignore_ascii_case("single")
            && !raw.trim().is_empty()
        {
            // An unknown / misspelled value falls back to `Single` for
            // safety. The operator almost certainly intended round-robin or
            // failover, so log a warning pointing at the typo.
            tracing::warn!(
                raw_mode = raw,
                resolved = "single",
                "unknown upstream.strategy.mode; expected one of \
                 `single`, `round_robin`, or `failover`. Defaulting to \
                 `single` (only the first [[upstream.providers]] entry is \
                 used). Fix the typo to enable the intended routing."
            );
        }
        parsed
    }

    fn from_str(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "round_robin" | "round-robin" | "roundrobin" => Self::RoundRobin,
            "failover" | "fail_over" => Self::Failover,
            // "single" and any unknown value default to Single so a typo
            // never silently enables an unintended strategy. The typo path
            // is logged by `from_str_logged`.
            _ => Self::Single,
        }
    }
}

/// Classify a connection-level `reqwest` error into the upstream error that
/// best matches its cause. Timeouts surface as `Timeout` (carrying the
/// configured duration so logs/dashboards read correctly); everything else
/// (DNS, connect, TLS) is `ConnectFailed`.
fn map_send_error(e: reqwest::Error, timeout_seconds: u64) -> UpstreamError {
    if e.is_timeout() {
        UpstreamError::Timeout(Duration::from_secs(timeout_seconds))
    } else {
        UpstreamError::ConnectFailed(e.to_string())
    }
}

/// Build a `HeaderName` from a user-supplied header string, defaulting to
/// `Authorization`. Unknown characters fall back to the canonical header so a
/// malformed config never panics at request time; the fallback is logged once
/// per call so operators notice a misconfigured `auth_header`.
fn safe_header_name(name: &str) -> HeaderName {
    match HeaderName::try_from(name) {
        Ok(n) => n,
        Err(_) => {
            tracing::warn!(
                auth_header = name,
                "invalid auth_header config; falling back to Authorization"
            );
            AUTHORIZATION
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str, url: &str) -> UpstreamProvider {
        UpstreamProvider {
            name: name.into(),
            url: url.into(),
            api_key: "ollama".into(),
            auth_header: "Authorization".into(),
            timeout_seconds: 1,
        }
    }

    #[test]
    fn build_headers_adds_bearer_authorization() {
        let upstream = ReqwestUpstream::new(provider("p", "http://127.0.0.1:1")).expect("build");
        let headers = upstream.build_headers();
        let auth = headers.get(AUTHORIZATION).expect("auth header present");
        assert_eq!(auth, "Bearer ollama");
    }

    #[test]
    fn build_headers_omits_authorization_when_api_key_empty() {
        let mut p = provider("p", "http://127.0.0.1:1");
        p.api_key = String::new();
        let upstream = ReqwestUpstream::new(p).expect("build");
        let headers = upstream.build_headers();
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn build_headers_uses_api_key_header_name_when_configured() {
        let mut p = provider("p", "http://127.0.0.1:1");
        p.auth_header = "api-key".into();
        let upstream = ReqwestUpstream::new(p).expect("build");
        let headers = upstream.build_headers();
        // Azure-style `api-key:` header carries the raw key (no Bearer prefix).
        let header = headers.get("api-key").expect("api-key header present");
        assert_eq!(header, "ollama");
    }

    #[test]
    fn new_rejects_api_key_with_invalid_header_bytes() {
        // Control characters are illegal in HTTP header values; the fail-fast
        // validation in `new` must surface this at construction time.
        let mut p = provider("p", "http://127.0.0.1:1");
        p.api_key = "bad\u{0000}key".into();
        match ReqwestUpstream::new(p) {
            Err(err) => assert!(
                err.to_string().contains("api_key"),
                "expected api_key in error: {err}"
            ),
            Ok(_) => panic!("expected construction to fail for an invalid api_key"),
        }
    }

    #[test]
    fn safe_header_name_falls_back_to_authorization_for_invalid_input() {
        assert_eq!(safe_header_name("not a valid header"), AUTHORIZATION);
        assert_eq!(safe_header_name("api-key"), "api-key");
    }

    // --- Pool construction ---------------------------------------------

    #[test]
    fn pool_construction_fails_when_a_provider_has_invalid_api_key() {
        let cfg = UpstreamConfig {
            providers: vec![
                provider("good", "http://x"),
                UpstreamProvider {
                    api_key: "bad\u{0000}key".into(),
                    ..provider("bad", "http://y")
                },
            ],
            ..UpstreamConfig::default()
        };
        match ReqwestUpstreamPool::new(&cfg) {
            Err(UpstreamError::ConnectFailed(msg)) => assert!(
                msg.contains("api_key"),
                "expected api_key failure, got: {msg}"
            ),
            other => panic!("expected ConnectFailed, got {other:?}"),
        }
    }

    #[test]
    fn pool_strategy_mode_parses_known_aliases() {
        assert_eq!(StrategyMode::from_str("single"), StrategyMode::Single);
        assert_eq!(
            StrategyMode::from_str("round_robin"),
            StrategyMode::RoundRobin
        );
        assert_eq!(
            StrategyMode::from_str("ROUND-ROBIN"),
            StrategyMode::RoundRobin
        );
        assert_eq!(StrategyMode::from_str("failover"), StrategyMode::Failover);
        // Unknown → safe default (Single).
        assert_eq!(StrategyMode::from_str("typo"), StrategyMode::Single);
        assert_eq!(StrategyMode::from_str(""), StrategyMode::Single);
    }

    // --- Behavioural routing tests (wiremock-backed) -------------------
    //
    // The routing logic (`single` / `round_robin` / `failover`) is the
    // core invariant of the multi-provider pool — these tests pin each
    // mode against N real wiremock HTTP servers so a regression in the
    // routing switch or the AtomicUsize counter is caught at the unit
    // level rather than only via a full e2e suite.

    use smos_application::types::ChatRequest;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::{body_partial_json, method};

    use crate::config::UpstreamStrategy;

    /// Build a `ChatRequest` small enough to satisfy the OpenAI shape the
    /// upstream forwards. We do not assert on the response body — only on
    /// which server received the call.
    fn probe_request() -> ChatRequest {
        let raw = serde_json::json!({
            "model": "probe",
            "messages": [{"role": "user", "content": "ping"}],
        });
        serde_json::from_value(raw).expect("probe ChatRequest")
    }

    /// Mount a 200-OK handler on `server` that responds with a unique
    /// JSON body so the test can tell which provider handled the call.
    /// Records each hit on the returned counter.
    async fn mount_ok(server: &MockServer, body: &'static str) -> Arc<AtomicUsize> {
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_for_closure = hits.clone();
        Mock::given(method("POST"))
            .and(body_partial_json(
                serde_json::json!({"messages": [{"role": "user"}]}),
            ))
            .respond_with(move |req: &wiremock::Request| {
                hits_for_closure.fetch_add(1, Ordering::SeqCst);
                let _ = req;
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"served_by": body}))
            })
            .mount(server)
            .await;
        hits
    }

    /// Mount a 500 handler that drives the `failover` strategy to the next
    /// provider. Records each failure hit on the returned counter so tests
    /// can assert the failing provider was actually consulted.
    async fn mount_500(server: &MockServer) -> Arc<AtomicUsize> {
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_for_closure = hits.clone();
        Mock::given(method("POST"))
            .respond_with(move |_: &wiremock::Request| {
                hits_for_closure.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(500).set_body_string("boom")
            })
            .mount(server)
            .await;
        hits
    }

    fn provider_for(server: &MockServer, name: &str) -> UpstreamProvider {
        UpstreamProvider {
            name: name.into(),
            url: format!("{}/v1/chat/completions", server.uri()),
            api_key: String::new(),
            auth_header: "Authorization".into(),
            timeout_seconds: 5,
        }
    }

    #[tokio::test]
    async fn pool_single_strategy_always_hits_first_provider() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        let h1 = mount_ok(&s1, "first").await;
        let h2 = mount_ok(&s2, "second").await;

        let cfg = UpstreamConfig {
            providers: vec![provider_for(&s1, "p1"), provider_for(&s2, "p2")],
            strategy: UpstreamStrategy {
                mode: "single".into(),
            },
        };
        let pool = ReqwestUpstreamPool::new(&cfg).expect("pool");

        for _ in 0..3 {
            let _ = pool.complete(probe_request()).await;
        }
        assert_eq!(h1.load(Ordering::SeqCst), 3);
        assert_eq!(h2.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn pool_round_robin_distributes_evenly_across_two_providers() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        let h1 = mount_ok(&s1, "first").await;
        let h2 = mount_ok(&s2, "second").await;

        let cfg = UpstreamConfig {
            providers: vec![provider_for(&s1, "p1"), provider_for(&s2, "p2")],
            strategy: UpstreamStrategy {
                mode: "round_robin".into(),
            },
        };
        let pool = ReqwestUpstreamPool::new(&cfg).expect("pool");

        for _ in 0..4 {
            let _ = pool.complete(probe_request()).await;
        }
        // 4 calls over 2 providers → 2 hits each.
        assert_eq!(h1.load(Ordering::SeqCst), 2);
        assert_eq!(h2.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn pool_failover_skips_failing_provider_and_returns_first_ok() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        let h1 = mount_500(&s1).await;
        let h2 = mount_ok(&s2, "second").await;

        let cfg = UpstreamConfig {
            providers: vec![provider_for(&s1, "p1"), provider_for(&s2, "p2")],
            strategy: UpstreamStrategy {
                mode: "failover".into(),
            },
        };
        let pool = ReqwestUpstreamPool::new(&cfg).expect("pool");

        let resp = pool.complete(probe_request()).await.expect("ok from p2");
        // The failing provider was consulted exactly once.
        assert_eq!(h1.load(Ordering::SeqCst), 1);
        // The healthy provider handled the call exactly once.
        assert_eq!(h2.load(Ordering::SeqCst), 1);
        // The non-streaming response body is the JSON returned by p2.
        match resp {
            ChatResponse::NonStreaming(v) => {
                assert_eq!(v["served_by"], serde_json::json!("second"));
            }
            _ => panic!("expected NonStreaming response"),
        }
    }

    #[tokio::test]
    async fn pool_failover_returns_all_providers_failed_when_every_provider_fails() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        let _h1 = mount_500(&s1).await;
        let _h2 = mount_500(&s2).await;

        let cfg = UpstreamConfig {
            providers: vec![provider_for(&s1, "p1"), provider_for(&s2, "p2")],
            strategy: UpstreamStrategy {
                mode: "failover".into(),
            },
        };
        let pool = ReqwestUpstreamPool::new(&cfg).expect("pool");

        match pool.complete(probe_request()).await {
            Err(UpstreamError::AllProvidersFailed(_)) => {}
            other => panic!("expected AllProvidersFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pool_single_strategy_with_single_provider_works() {
        // Sanity check: the pool does not regress the single-provider
        // shape under `single`.
        let s1 = MockServer::start().await;
        let h1 = mount_ok(&s1, "only").await;

        let cfg = UpstreamConfig {
            providers: vec![provider_for(&s1, "only")],
            strategy: UpstreamStrategy {
                mode: "single".into(),
            },
        };
        let pool = ReqwestUpstreamPool::new(&cfg).expect("pool");
        let _ = pool.complete(probe_request()).await.expect("ok");
        assert_eq!(h1.load(Ordering::SeqCst), 1);
    }
}
