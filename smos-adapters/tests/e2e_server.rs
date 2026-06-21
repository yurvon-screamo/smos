//! E2E: health probe, CORS headers, upstream error propagation.

mod common;

use common::spawn_smos;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn health_returns_ok_with_version() {
    let upstream = MockServer::start().await;
    let smos = spawn_smos(&upstream.uri()).await;

    let resp = reqwest::Client::new()
        .get(format!("{smos}/health"))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);
    let body = resp.json::<serde_json::Value>().await.expect("json");
    assert_eq!(body["status"], "ok");
    assert!(body["version"].is_string());
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn cors_allow_origin_header_is_present_on_chat() {
    let upstream = MockServer::start().await;
    // A bare mock so the request does not hang awaiting a real upstream.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })))
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .header("origin", "https://example.com")
        .header("access-control-request-method", "POST")
        .json(&json!({"model": "m", "messages": []}))
        .send()
        .await
        .expect("send");
    let headers = resp.headers();
    assert_eq!(
        headers
            .get("access-control-allow-origin")
            .map(|v| v.to_str().unwrap_or("")),
        Some("*")
    );
}

#[tokio::test]
async fn upstream_500_propagates_as_502_bad_gateway() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream boom"))
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&json!({"model": "m", "messages": []}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 502);
}
