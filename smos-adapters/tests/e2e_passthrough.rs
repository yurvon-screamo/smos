//! E2E: streaming passthrough + session marker injection.

mod common;

use common::{SSE_HELLO_WORLD, chat_body, session_id_in, spawn_smos, sse_payloads};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn streaming_upstream(server: &MockServer, body: &'static str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("content-type", "application/json"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(server)
        .await;
}

#[tokio::test]
async fn streaming_passthrough_forwards_every_chunk() {
    let upstream = MockServer::start().await;
    streaming_upstream(&upstream, SSE_HELLO_WORLD).await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("origa:gpt-4o", vec![("stream", json!(true))]);

    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);
    let raw = resp.text().await.expect("body");
    let payloads = sse_payloads(&raw);

    // "Hello" and " world" pass through verbatim.
    assert_eq!(payloads.len(), 4);
    assert!(payloads[0].contains("Hello"));
    assert!(payloads[1].contains(" world"));
}

#[tokio::test]
async fn streaming_appends_marker_to_terminal_stop_chunk() {
    let upstream = MockServer::start().await;
    streaming_upstream(&upstream, SSE_HELLO_WORLD).await;

    let smos = spawn_smos(&upstream.uri()).await;
    let body = chat_body("origa:gpt-4o", vec![("stream", json!(true))]);

    let raw = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send")
        .text()
        .await
        .expect("body");
    let payloads = sse_payloads(&raw);

    // The stop chunk (payload[2]) must carry the marker in delta.content.
    let stop: serde_json::Value = serde_json::from_str(&payloads[2]).expect("stop json");
    let content = stop["choices"][0]["delta"]["content"]
        .as_str()
        .expect("content");
    let id = session_id_in(content).expect("marker present in stop chunk");
    assert!(id.starts_with("sess_"));
    assert_eq!(id.len(), "sess_".len() + 12);

    // [DONE] is still the final frame.
    assert_eq!(payloads[3], "[DONE]");
}

#[tokio::test]
async fn streaming_marker_uses_new_session_when_history_has_none() {
    let upstream = MockServer::start().await;
    streaming_upstream(&upstream, SSE_HELLO_WORLD).await;

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
    let id = session_id_in(&raw).expect("marker");
    assert!(id.starts_with("sess_") && id.len() == 17);
}

#[tokio::test]
async fn streaming_reuses_session_id_from_history() {
    let upstream = MockServer::start().await;
    streaming_upstream(&upstream, SSE_HELLO_WORLD).await;

    let smos = spawn_smos(&upstream.uri()).await;
    // History carries an existing marker the proxy must detect and reuse.
    let body = json!({
        "model": "k:m",
        "stream": true,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi back\n<!-- smos:sess_aaaaaaaaaaaa -->"},
            {"role": "user", "content": "again"},
        ],
    });

    let raw = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send")
        .text()
        .await
        .expect("body");
    let id = session_id_in(&raw).expect("marker");
    assert_eq!(id, "sess_aaaaaaaaaaaa");
}

#[tokio::test]
async fn streaming_marker_still_emitted_when_upstream_closes_without_done() {
    // Upstream sends a stop chunk but no [DONE], then closes the stream.
    let truncated = "\
data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\
\n\
data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";

    let upstream = MockServer::start().await;
    streaming_upstream(&upstream, truncated).await;

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
    // The stop chunk still received the marker even though [DONE] was absent.
    assert!(session_id_in(&raw).is_some());
}

#[tokio::test]
async fn streaming_multiple_content_chunks_keep_order() {
    let body_text = "\
data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A\"},\"finish_reason\":null}]}\n\
\n\
data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"B\"},\"finish_reason\":null}]}\n\
\n\
data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"C\"},\"finish_reason\":null}]}\n\
\n\
data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";

    let upstream = MockServer::start().await;
    streaming_upstream(&upstream, body_text).await;

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
    let payloads = sse_payloads(&raw);
    assert_eq!(payloads.len(), 4);
    assert!(payloads[0].contains("\"A\""));
    assert!(payloads[1].contains("\"B\""));
    assert!(payloads[2].contains("\"C\""));
    // stop chunk (payloads[3]) carries the marker.
    assert!(session_id_in(&payloads[3]).is_some());
}
