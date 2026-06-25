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
//! # Multi-provider router
//!
//! [`ReqwestUpstreamRouter`] wraps N [`ReqwestUpstream`] instances keyed by
//! provider name. The routing decision is made by the application layer's
//! `route_request` helper (which resolves a `[persons.X]` entry to a
//! concrete provider name); the router then looks the name up in its
//! internal map and forwards the request.
//!
//! Unknown provider names surface as
//! [`UpstreamError::ConnectFailed`] so the HTTP layer maps them to 502 —
//! this should be unreachable in practice because the application layer
//! validates the provider reference at startup.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use smos_application::errors::UpstreamError;
use smos_application::ports::LlmUpstream;
use smos_application::types::{ChatRequest, ChatResponse};

use crate::config::ProviderConfig;

/// HTTP upstream backed by a pooled `reqwest::Client`.
#[derive(Clone)]
pub struct ReqwestUpstream {
    client: Client,
    inner: Arc<UpstreamInner>,
}

#[derive(Debug)]
struct UpstreamInner {
    name: String,
    url: String,
    api_key: String,
    auth_header: String,
    /// Configured request timeout. Carried into the inner struct so the
    /// send-error path can surface the real value in `UpstreamError::Timeout`
    /// instead of a misleading `0s` placeholder.
    timeout: Duration,
}

impl fmt::Debug for ReqwestUpstream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestUpstream")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl ReqwestUpstream {
    /// Build a new upstream from a [`ProviderConfig`]. The api-key is read
    /// from the env var named in `api_key_env` (resolved through
    /// [`ProviderConfig::resolve_api_key`]). Validates the resolved key (if
    /// non-empty) up front so a misconfigured secret with control characters
    /// fails fast at startup rather than silently producing an
    /// unauthenticated request later.
    pub fn from_config(provider: &ProviderConfig) -> Result<Self, UpstreamError> {
        let api_key = provider.resolve_api_key();
        Self::new(
            &provider.name,
            &provider.url,
            &api_key,
            &provider.auth_header,
            provider.timeout_seconds,
        )
    }

    /// Build a new upstream from explicit parameters. Used by tests and by
    /// [`ReqwestUpstreamRouter::from_config`].
    pub fn new(
        name: &str,
        url: &str,
        api_key: &str,
        auth_header: &str,
        timeout_seconds: u64,
    ) -> Result<Self, UpstreamError> {
        if !api_key.is_empty()
            && let Err(e) = HeaderValue::from_str(api_key)
        {
            return Err(UpstreamError::ConnectFailed(format!(
                "provider {name:?}: api_key contains invalid header bytes: {e}"
            )));
        }
        let timeout = Duration::from_secs(timeout_seconds);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| UpstreamError::ConnectFailed(e.to_string()))?;
        Ok(Self {
            client,
            inner: Arc::new(UpstreamInner {
                name: name.into(),
                url: url.into(),
                api_key: api_key.into(),
                auth_header: auth_header.into(),
                timeout,
            }),
        })
    }

    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if !self.inner.api_key.is_empty() {
            // Azure-style `api-key:` header carries the raw key; every other
            // header name uses the `Authorization: Bearer <key>` scheme.
            let is_api_key_header = self.inner.auth_header.eq_ignore_ascii_case("api-key");
            let value = if is_api_key_header {
                HeaderValue::from_str(&self.inner.api_key)
            } else {
                HeaderValue::from_str(&format!("Bearer {}", self.inner.api_key))
            }
            .expect("api_key validated at construction");
            let name = safe_header_name(&self.inner.auth_header);
            headers.insert(name, value);
        }
        headers
    }

    fn provider_name(&self) -> &str {
        &self.inner.name
    }

    async fn send(&self, request: &ChatRequest) -> Result<reqwest::Response, UpstreamError> {
        let body = serde_json::to_value(request)
            .map_err(|e| UpstreamError::SerializationError(e.to_string()))?;
        let response = self
            .client
            .post(&self.inner.url)
            .headers(self.build_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| map_send_error(e, self.inner.url.as_str(), self.inner.timeout))?;
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
    async fn complete(
        &self,
        _provider_name: &str,
        request: ChatRequest,
    ) -> Result<ChatResponse, UpstreamError> {
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

/// Router of [`ReqwestUpstream`] instances backing a single [`LlmUpstream`]
/// trait surface. Each request is routed to the provider named by the
/// caller (resolved upstream by `route_request` against the `[persons.*]`
/// map).
///
/// Construction fails fast if any provider's resolved api-key carries
/// invalid header bytes — partial construction would leave a router with
/// N-1 working providers and one silently broken, which is the exact
/// failure mode the multi-provider design is supposed to surface up front.
///
/// Cheap to clone: every clone shares the same inner state (providers map)
/// via `Arc`. The HTTP-side fan-out therefore observes a single global map
/// across all router clones, which matches the operator's mental model of
/// one "logical" upstream even though axum hands each request a fresh
/// [`AppState`](crate::http::axum_server::AppState) snapshot.
#[derive(Clone, Debug)]
pub struct ReqwestUpstreamRouter {
    inner: Arc<RouterInner>,
}

#[derive(Debug)]
struct RouterInner {
    /// Provider name → upstream. Indexed by name so the lookup is O(1) and
    /// the adapter does not need to linear-scan the providers list on every
    /// chat-completion request.
    by_name: HashMap<String, ReqwestUpstream>,
}

impl ReqwestUpstreamRouter {
    /// Build a router from a slice of [`ProviderConfig`]. The router takes
    /// ownership of its [`ReqwestUpstream`] instances so the config struct
    /// can be dropped after wiring.
    ///
    /// Duplicate provider names are rejected with
    /// [`UpstreamError::ConnectFailed`] — the config validator already
    /// catches duplicates at startup, this is the defensive second line.
    pub fn from_config(providers: &[ProviderConfig]) -> Result<Self, UpstreamError> {
        if providers.is_empty() {
            return Err(UpstreamError::ConnectFailed(
                "no upstream providers configured".into(),
            ));
        }
        let mut by_name: HashMap<String, ReqwestUpstream> = HashMap::new();
        for (idx, provider) in providers.iter().enumerate() {
            // Surface the offending provider's index + name in the error
            // chain so an operator with N configured providers can tell
            // which entry has the bad api_key. Without this annotation the
            // underlying `UpstreamError::ConnectFailed` only carries the
            // generic "api_key contains invalid header bytes" message.
            let provider_name = provider.name.clone();
            let upstream = ReqwestUpstream::from_config(provider).map_err(|e| {
                UpstreamError::ConnectFailed(format!(
                    "provider[{idx}] (name={provider_name:?}) rejected: {e}"
                ))
            })?;
            if by_name.insert(provider.name.clone(), upstream).is_some() {
                return Err(UpstreamError::ConnectFailed(format!(
                    "provider[{idx}] (name={provider_name:?}): duplicate provider name"
                )));
            }
        }
        Ok(Self {
            inner: Arc::new(RouterInner { by_name }),
        })
    }

    /// Build a router from a pre-constructed `Vec<ReqwestUpstream>`. Used by
    /// tests that need to wire mocks without going through the
    /// `ProviderConfig` indirection.
    pub fn from_upstreams(upstreams: Vec<ReqwestUpstream>) -> Result<Self, UpstreamError> {
        if upstreams.is_empty() {
            return Err(UpstreamError::ConnectFailed(
                "no upstream providers configured".into(),
            ));
        }
        let mut by_name: HashMap<String, ReqwestUpstream> = HashMap::new();
        for upstream in upstreams {
            let name = upstream.provider_name().to_string();
            if by_name.insert(name.clone(), upstream).is_some() {
                return Err(UpstreamError::ConnectFailed(format!(
                    "duplicate provider name: {name:?}"
                )));
            }
        }
        Ok(Self {
            inner: Arc::new(RouterInner { by_name }),
        })
    }

    /// Number of providers in the router. Exposed for diagnostics / tests.
    pub fn provider_count(&self) -> usize {
        self.inner.by_name.len()
    }

    /// Look up the provider by name (used by tests + diagnostics).
    pub fn provider(&self, name: &str) -> Option<&ReqwestUpstream> {
        self.inner.by_name.get(name)
    }
}

impl LlmUpstream for ReqwestUpstreamRouter {
    async fn complete(
        &self,
        provider_name: &str,
        request: ChatRequest,
    ) -> Result<ChatResponse, UpstreamError> {
        let upstream = self.inner.by_name.get(provider_name).ok_or_else(|| {
            UpstreamError::ConnectFailed(format!(
                "unknown provider {provider_name:?}; not in [[providers]] list"
            ))
        })?;
        upstream.complete(provider_name, request).await
    }
}

/// Classify a connection-level `reqwest` error into the upstream error that
/// best matches its cause. Timeouts surface as `Timeout` carrying the
/// configured duration so logs / dashboards / `Display` read correctly;
/// everything else (DNS, connect, TLS) is `ConnectFailed` (carrying the URL
/// so an operator with N providers can identify which one is unreachable).
fn map_send_error(e: reqwest::Error, url: &str, timeout: Duration) -> UpstreamError {
    if e.is_timeout() {
        UpstreamError::Timeout(timeout)
    } else {
        UpstreamError::ConnectFailed(format!("url={url}: {e}"))
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

    fn provider(name: &str, url: &str) -> ProviderConfig {
        ProviderConfig {
            name: name.into(),
            url: url.into(),
            api_key_env: String::new(),
            auth_header: "Authorization".into(),
            timeout_seconds: 1,
        }
    }

    /// Helper: construct an `UpstreamInner`-backed `ReqwestUpstream` with
    /// a literal api_key (bypasses env-var resolution so the test does not
    /// depend on the environment).
    fn upstream_with_key(name: &str, url: &str, api_key: &str) -> ReqwestUpstream {
        ReqwestUpstream::new(name, url, api_key, "Authorization", 1).expect("build")
    }

    #[test]
    fn build_headers_adds_bearer_authorization() {
        let upstream = upstream_with_key("p", "http://127.0.0.1:1", "sample-key");
        let headers = upstream.build_headers();
        let auth = headers.get(AUTHORIZATION).expect("auth header present");
        assert_eq!(auth, "Bearer sample-key");
    }

    #[test]
    fn build_headers_omits_authorization_when_api_key_empty() {
        let upstream = upstream_with_key("p", "http://127.0.0.1:1", "");
        let headers = upstream.build_headers();
        assert!(headers.get(AUTHORIZATION).is_none());
    }

    #[test]
    fn build_headers_uses_api_key_header_name_when_configured() {
        let upstream = ReqwestUpstream::new("p", "http://127.0.0.1:1", "sample-key", "api-key", 1)
            .expect("build");
        let headers = upstream.build_headers();
        // Azure-style `api-key:` header carries the raw key (no Bearer prefix).
        let header = headers.get("api-key").expect("api-key header present");
        assert_eq!(header, "sample-key");
    }

    #[test]
    fn new_rejects_api_key_with_invalid_header_bytes() {
        // Control characters are illegal in HTTP header values; the fail-fast
        // validation in `new` must surface this at construction time.
        match ReqwestUpstream::new(
            "p",
            "http://127.0.0.1:1",
            "bad\u{0000}key",
            "Authorization",
            1,
        ) {
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

    // --- Router construction -----------------------------------------

    #[test]
    fn router_construction_fails_when_a_provider_has_invalid_api_key() {
        // We use the from_upstreams path so we can inject a bad key directly
        // (ProviderConfig::resolve_api_key would otherwise return empty for
        // an unconfigured api_key_env).
        let good = upstream_with_key("good", "http://x", "k");
        let bad = ReqwestUpstream::new("bad", "http://y", "bad\u{0000}key", "Authorization", 1)
            .expect_err("bad key rejected at construction");
        // The bad upstream cannot even be built; assert the error mentions
        // the api_key field.
        assert!(bad.to_string().contains("api_key"));
        // Sanity: the good upstream alone builds a router.
        let router = ReqwestUpstreamRouter::from_upstreams(vec![good]).expect("router");
        assert_eq!(router.provider_count(), 1);
    }

    #[test]
    fn router_construction_rejects_duplicate_provider_names() {
        let a = upstream_with_key("dup", "http://a", "k1");
        let b = upstream_with_key("dup", "http://b", "k2");
        match ReqwestUpstreamRouter::from_upstreams(vec![a, b]) {
            Err(UpstreamError::ConnectFailed(msg)) => assert!(
                msg.contains("duplicate"),
                "expected duplicate message, got: {msg}"
            ),
            other => panic!("expected ConnectFailed, got {other:?}"),
        }
    }

    #[test]
    fn router_construction_rejects_empty_provider_list() {
        let err = ReqwestUpstreamRouter::from_upstreams(vec![]).expect_err("empty");
        assert!(err.to_string().contains("no upstream providers"));
    }

    #[test]
    fn router_from_config_resolves_api_key_from_env() {
        let _guard = crate::test_env_lock::lock();
        let prior = std::env::var("SMOS_TEST_ROUTER_KEY").ok();
        // SAFETY: the workspace env-test lock is held for the duration
        // of the env mutation + read, and the prior value is restored
        // before return so other tests in the binary see the original
        // state.
        unsafe {
            std::env::set_var("SMOS_TEST_ROUTER_KEY", "sk-from-env");
        }
        let mut provider = provider("p", "http://x");
        provider.api_key_env = "SMOS_TEST_ROUTER_KEY".into();
        let router = ReqwestUpstreamRouter::from_config(&[provider]).expect("router");
        // SAFETY: same single-file serialisation guarantee.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_TEST_ROUTER_KEY", v),
                None => std::env::remove_var("SMOS_TEST_ROUTER_KEY"),
            }
        }
        assert_eq!(router.provider_count(), 1);
    }

    // --- Behavioural routing tests (wiremock-backed) -----------------
    //
    // The router routes each request to the named provider. These tests
    // pin the per-name lookup against real wiremock HTTP servers so a
    // regression in the HashMap is caught at the unit level rather than
    // only via a full e2e suite.

    use smos_application::types::ChatRequest;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::{body_partial_json, method};

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
    async fn mount_ok(server: &MockServer, body: &'static str) {
        Mock::given(method("POST"))
            .and(body_partial_json(
                serde_json::json!({"messages": [{"role": "user"}]}),
            ))
            .respond_with(move |_: &wiremock::Request| {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"served_by": body}))
            })
            .mount(server)
            .await;
    }

    fn upstream_for(server: &MockServer, name: &str) -> ReqwestUpstream {
        ReqwestUpstream::new(
            name,
            &format!("{}/v1/chat/completions", server.uri()),
            "",
            "Authorization",
            5,
        )
        .expect("upstream")
    }

    #[tokio::test]
    async fn router_routes_request_to_named_provider() {
        let s1 = MockServer::start().await;
        let s2 = MockServer::start().await;
        mount_ok(&s1, "first").await;
        mount_ok(&s2, "second").await;

        let router = ReqwestUpstreamRouter::from_upstreams(vec![
            upstream_for(&s1, "p1"),
            upstream_for(&s2, "p2"),
        ])
        .expect("router");

        // Route to p2 — the request MUST land on s2, not s1.
        let resp = router.complete("p2", probe_request()).await.expect("ok");
        match resp {
            ChatResponse::NonStreaming(v) => {
                assert_eq!(v["served_by"], serde_json::json!("second"));
            }
            _ => panic!("expected NonStreaming response"),
        }
    }

    #[tokio::test]
    async fn router_unknown_provider_returns_connect_failed() {
        let s1 = MockServer::start().await;
        mount_ok(&s1, "first").await;

        let router =
            ReqwestUpstreamRouter::from_upstreams(vec![upstream_for(&s1, "p1")]).expect("router");

        match router.complete("ghost", probe_request()).await {
            Err(UpstreamError::ConnectFailed(msg)) => {
                assert!(msg.contains("ghost"), "msg = {msg}");
            }
            other => panic!("expected ConnectFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn router_with_single_provider_works() {
        let s1 = MockServer::start().await;
        mount_ok(&s1, "only").await;

        let router =
            ReqwestUpstreamRouter::from_upstreams(vec![upstream_for(&s1, "only")]).expect("router");
        let _ = router.complete("only", probe_request()).await.expect("ok");
    }
}
