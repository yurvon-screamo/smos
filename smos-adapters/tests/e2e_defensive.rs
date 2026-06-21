//! E2E: defensive branches + charset correctness.

mod common;

use common::{chat_body, session_id_in, spawn_smos};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn streaming_emits_synthetic_marker_when_no_stop_and_no_done() {
    // Upstream sends only content chunks, then closes — no stop event, no
    // [DONE]. The proxy must still inject the session marker via its safety
    // net so the client receives the trailer for continuity.
    let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("m", vec![("stream", json!(true))]);
    let raw = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send")
        .text()
        .await
        .expect("body");
    // The synthetic marker chunk must carry the marker even though the
    // upstream never produced a finish_reason=stop event.
    assert!(
        session_id_in(&raw).is_some(),
        "expected a synthetic session marker in the stream: {raw}"
    );
}

#[tokio::test]
async fn streaming_passthrough_preserves_cyrillic() {
    // E2E sanity that Cyrillic survives the full HTTP round-trip. The split-
    // across-chunks corruption case (the actual byte-buffering regression) is
    // covered by the byte-by-byte unit tests in sse_parser, since wiremock
    // delivers this short body in a single chunk.
    let cyrillic = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Привет, мир\"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("content-type", "text/event-stream")
                .set_body_string(cyrillic),
        )
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("m", vec![("stream", json!(true))]);
    let raw = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send")
        .text()
        .await
        .expect("body");
    // Cyrillic must survive the byte-buffered passthrough undistorted.
    assert!(
        raw.contains("Привет, мир"),
        "multibyte content corrupted: {raw}"
    );
}

#[tokio::test]
async fn streaming_appends_marker_to_tool_calls_terminal_chunk() {
    // Function-calling conversations end with finish_reason="tool_calls", not
    // "stop". The marker must still land on that terminal chunk so the session
    // is recoverable on the next request.
    let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n\
data: [DONE]\n\n";

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("m", vec![("stream", json!(true))]);
    let raw = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send")
        .text()
        .await
        .expect("body");
    assert!(
        session_id_in(&raw).is_some(),
        "marker missing on tool_calls terminal chunk: {raw}"
    );
}

#[tokio::test]
async fn unreachable_upstream_returns_502() {
    // Point SMOS at a port where nothing listens — connect fails → 502.
    let smos = spawn_smos("http://127.0.0.1:1").await;
    let body = chat_body("m", vec![]);
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 502);
}

#[tokio::test]
async fn upstream_4xx_body_is_propagated_verbatim() {
    // The original OpenAI-shaped error structure must reach the client
    // unchanged (not re-wrapped in an SMOS envelope).
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string(
            r#"{"error":{"message":"bad","type":"invalid_request_error","code":"x"}}"#,
        ))
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("m", vec![]);
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
    let payload = resp.json::<serde_json::Value>().await.expect("json");
    // The upstream's structured error survives verbatim (no SMOS wrapper).
    assert_eq!(payload["error"]["type"], "invalid_request_error");
    assert_eq!(payload["error"]["code"], "x");
}

#[tokio::test]
async fn upstream_5xx_body_does_not_leak_to_client() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500).set_body_string("INTERNAL: secret stacktrace /etc/secrets"),
        )
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("m", vec![]);
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 502);
    let payload = resp.json::<serde_json::Value>().await.expect("json");
    let message = payload["error"]["message"].as_str().expect("message");
    assert!(!message.contains("stacktrace"));
    assert!(!message.contains("/etc/secrets"));
}
