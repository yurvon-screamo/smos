//! E2E: non-streaming marker injection, person routing, session detection.

mod common;

use common::{TEST_PERSON, chat_body, session_id_in, spawn_smos};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn json_upstream(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop",
            }],
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn non_streaming_appends_marker_to_message_content() {
    let upstream = MockServer::start().await;
    json_upstream(&upstream).await;

    let smos = spawn_smos(&upstream.uri()).await;
    // The test fixture configures `[persons.origa]`; the client sends the
    // person name `origa` and the proxy rewrites `request.model` to the
    // upstream model declared in the person entry.
    let body = chat_body(TEST_PERSON, vec![]);

    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send")
        .json::<serde_json::Value>()
        .await
        .expect("json");

    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert!(content.starts_with("hi there"));
    assert!(content.contains("<!-- smos:sess_"));
    let id = session_id_in(content).expect("marker");
    assert!(id.starts_with("sess_") && id.len() == 17);
}

#[tokio::test]
async fn person_routing_rewrites_request_model_to_upstream_model() {
    let upstream = MockServer::start().await;
    // The mock only matches when the forwarded body carries the upstream
    // model declared in `[persons.origa].model` (`gpt-4o`). A regression in
    // the routing layer that forwards the raw person name would 404 the
    // mock and the test would fail.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "gpt-4o"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })))
        .expect(1)
        .mount(&upstream)
        .await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body(TEST_PERSON, vec![]);
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn unknown_person_is_rejected_with_400() {
    // The proxy no longer falls back to "no prefix → forward as-is". A
    // model that does not match any `[persons.*]` entry is rejected at
    // the routing layer with a 400 Bad Request pointing at the missing
    // person — so the operator gets a clear error instead of a confusing
    // 404 from the upstream.
    let upstream = MockServer::start().await;
    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("not-a-person", vec![]);

    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
    let text = resp.text().await.expect("body");
    assert!(
        text.contains("unknown person"),
        "expected body to mention 'unknown person', got: {text}"
    );
}

#[tokio::test]
async fn unsafe_memory_key_is_rejected_with_400() {
    let upstream = MockServer::start().await;
    let smos = spawn_smos(&upstream.uri()).await;
    // Path-traversal characters in a person name still fail MemoryKey
    // validation, even before the persons map is consulted.
    let body = chat_body("../etc", vec![]);

    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
    let text = resp.text().await.expect("body");
    assert!(
        text.contains("memory key"),
        "expected body to mention 'memory key', got: {text}"
    );
}

#[tokio::test]
async fn session_detection_picks_up_marker_in_multipart_history() {
    let upstream = MockServer::start().await;
    json_upstream(&upstream).await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = json!({
        "model": TEST_PERSON,
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "prev\n<!-- smos:sess_bbbbbbbbbbbb -->"},
            ],
        }],
    });

    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    let id = session_id_in(content).expect("marker");
    assert_eq!(id, "sess_bbbbbbbbbbbb");
}
