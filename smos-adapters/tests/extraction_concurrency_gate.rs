//! Gate-contract test for `OllamaExtractor::with_slot`.
//!
//! The serve path shares a single-slot upstream (`llama-server -np 1`) between
//! chat-completion forwards and background fact-extraction. Without a gate, N
//! concurrent extraction HTTP calls pile up on the slot and starve forwards.
//! `with_slot(Arc<Semaphore>::new(max))` bounds the in-flight extraction calls
//! to `max` so chat forwards wait behind at most `max` extractions, not the
//! whole backlog.
//!
//! This test pins that contract: with a 1-permit gate, at most 1 extraction
//! call is in flight to the HTTP server at any instant.
//!
//! # Why a hand-rolled TCP server instead of wiremock
//!
//! wiremock serialises mock matching under an internal lock, so a responder
//! that sleeps never observes real overlap — peak concurrency is always 1
//! regardless of the client. To prove the gate actually bounds *concurrent*
//! in-flight calls, the fake upstream spawns one task per accepted
//! connection (`tokio::spawn`), so requests genuinely overlap when the client
//! issues them in parallel.

#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use smos::OllamaExtractor;
use smos::config::LlmExtractionConfig;
use smos_application::ports::LlmExtractor;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

/// Minimal HTTP/1.1 upstream that tracks peak concurrency. Each accepted
/// connection is served by its own task, so simultaneous connections overlap
/// (the basis for observing real concurrency). The handler:
///   1. reads the request head + body (Content-Length bytes),
///   2. bumps `in_flight` and a `max_seen` high-water mark,
///   3. sleeps `hold` so peers overlap,
///   4. drops `in_flight`,
///   5. writes a 200 response carrying the OpenAI chat-completion shape the
///      extractor deserialises.
///
/// `Connection: close` keeps the connection lifecycle one-request-per-socket,
/// so each parallel `extract_facts` opens its own connection and gets its own
/// task — exactly the concurrency we want to observe.
struct ConcurrencyUpstream {
    in_flight: Arc<AtomicUsize>,
    max_seen: Arc<AtomicUsize>,
}

impl ConcurrencyUpstream {
    fn new() -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let me = ConcurrencyUpstream {
            in_flight: in_flight.clone(),
            max_seen: max_seen.clone(),
        };
        (me, in_flight, max_seen)
    }

    async fn serve(self, listener: TcpListener, hold: Duration) {
        loop {
            let (mut stream, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => continue,
            };
            let in_flight = self.in_flight.clone();
            let max_seen = self.max_seen.clone();
            tokio::spawn(async move {
                if read_request(&mut stream).await.is_err() {
                    return;
                }
                let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                bump_max(&max_seen, current);
                tokio::time::sleep(hold).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                let _ = write_chat_response(&mut stream).await;
            });
        }
    }
}

/// Read one HTTP/1.1 request: the head up to `\r\n\r\n`, parse
/// `Content-Length`, then read exactly that many body bytes. Anything
/// malformed is ignored (the test only cares the connection was served).
async fn read_request(stream: &mut tokio::net::TcpStream) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(512);
    let mut byte = [0u8; 1];
    loop {
        let n = stream.read(&mut byte).await?;
        if n == 0 {
            return Ok(());
        }
        buf.push(byte[0]);
        if buf.ends_with(b"\r\n\r\n") {
            break;
        }
        if buf.len() > 8 * 1024 {
            return Ok(());
        }
    }
    let content_length = extract_content_length(&buf);
    if content_length > 0 {
        let mut body = vec![0u8; content_length];
        stream.read_exact(&mut body).await?;
    }
    Ok(())
}

fn extract_content_length(head: &[u8]) -> usize {
    let text = String::from_utf8_lossy(head);
    for line in text.split("\r\n") {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:")
            && let Ok(n) = rest.trim().parse::<usize>()
        {
            return n;
        }
    }
    0
}

async fn write_chat_response(stream: &mut tokio::net::TcpStream) -> std::io::Result<()> {
    let body = serde_json::json!({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "- a single extracted fact"},
            "finish_reason": "stop",
        }],
    });
    let body = serde_json::to_vec(&body).expect("serialize body");
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.flush().await?;
    Ok(())
}

fn bump_max(max: &Arc<AtomicUsize>, candidate: usize) {
    let mut current = max.load(Ordering::SeqCst);
    while candidate > current {
        match max.compare_exchange(current, candidate, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return,
            Err(actual) => current = actual,
        }
    }
}

async fn spawn_upstream(hold: Duration) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let (upstream, _in_flight, max_seen) = ConcurrencyUpstream::new();
    tokio::spawn(async move {
        upstream.serve(listener, hold).await;
    });
    (format!("http://{addr}"), max_seen)
}

fn gated_extractor(url: String, permits: usize) -> OllamaExtractor {
    let cfg = Arc::new(LlmExtractionConfig {
        url,
        timeout_seconds: 15,
        ..LlmExtractionConfig::default()
    });
    OllamaExtractor::new(cfg)
        .expect("extractor")
        .with_slot(Arc::new(Semaphore::new(permits)), permits)
}

async fn fire_concurrent(extractor: &OllamaExtractor, count: usize) {
    let mut handles = Vec::with_capacity(count);
    for _ in 0..count {
        let ext = extractor.clone();
        handles.push(tokio::spawn(async move {
            let _ = ext
                .extract_facts("assistant reply long enough to pass the gate", &[])
                .await;
        }));
    }
    for handle in handles {
        let _ = handle.await;
    }
}

/// With a 1-permit gate, at most 1 extraction call is in flight at once.
/// Before the gate acquire is wired into `extract_facts`, the 8 parallel calls
/// all overlap on the upstream → `max_seen` climbs well past 1 → assertion
/// fails (RED). Once wired, only 1 runs at a time → `max_seen == 1` (GREEN).
///
/// `multi_thread` guarantees real task overlap (the upstream sleeps, so the
/// client calls must overlap for the assertion to be meaningful) regardless of
/// CI load — a current-thread runtime serialises spawned-task progress tightly
/// enough under contention that the overlap could thin out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_permit_gate_bounds_concurrent_extraction_calls_to_one() {
    let (url, max_seen) = spawn_upstream(Duration::from_millis(80)).await;
    let extractor = gated_extractor(url, 1);
    fire_concurrent(&extractor, 8).await;

    let observed = max_seen.load(Ordering::SeqCst);
    assert!(
        observed <= 1,
        "a 1-permit extraction gate must allow at most 1 concurrent in-flight call, \
         observed peak concurrency = {observed}"
    );
}

/// A wider gate (3 permits) allows up to 3 concurrent calls — proving the gate
/// size tracks the semaphore, not a hardcoded 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn three_permit_gate_allows_up_to_three_concurrent_calls() {
    let (url, max_seen) = spawn_upstream(Duration::from_millis(80)).await;
    let extractor = gated_extractor(url, 3);
    fire_concurrent(&extractor, 8).await;

    let observed = max_seen.load(Ordering::SeqCst);
    assert!(
        observed <= 3,
        "a 3-permit extraction gate must allow at most 3 concurrent in-flight calls, \
         observed peak concurrency = {observed}"
    );
    assert!(
        observed >= 2,
        "a 3-permit gate with 8 concurrent callers should observe at least 2 overlapping, \
         observed peak concurrency = {observed} (gate may be inert)"
    );
}
