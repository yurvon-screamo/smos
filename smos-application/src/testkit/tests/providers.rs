//! ScriptedNliClassifier + ScriptedExtractor + ConstantEmbedder + FixedClock
//! + ScriptedReranker parity invariants.

use super::*;
use crate::ports::RerankProvider;
use crate::types::RerankResult;

#[tokio::test]
async fn scripted_returns_in_order() {
    let v1 = neutral_verdict();
    let v2 = NliResult {
        label: NliLabel::Entailment,
        scores: NliScores {
            entailment: 0.9,
            neutral: 0.08,
            contradiction: 0.02,
        },
        available: true,
    };
    let classifier = ScriptedNliClassifier::new(vec![Ok(v1.clone()), Ok(v2.clone())]);

    let r1 = classifier.classify("p1", "h1").await.unwrap();
    let r2 = classifier.classify("p2", "h2").await.unwrap();
    let r3 = classifier.classify("p3", "h3").await;

    assert_eq!(r1.label, v1.label);
    assert_eq!(r2.label, v2.label);
    assert!(r3.is_err(), "exhausted queue must error");
}

#[tokio::test]
async fn matching_invoked_per_call() {
    let classifier = ScriptedNliClassifier::matching(|_p, _h| Ok(neutral_verdict()));

    classifier
        .classify("premise A", "hypothesis A")
        .await
        .unwrap();
    classifier
        .classify("premise B", "hypothesis B")
        .await
        .unwrap();

    let calls = classifier.calls();
    assert_eq!(calls.len(), 2, "closure invoked once per call");
    assert_eq!(
        calls[0],
        ("premise A".to_string(), "hypothesis A".to_string())
    );
    assert_eq!(
        calls[1],
        ("premise B".to_string(), "hypothesis B".to_string())
    );
}

#[tokio::test]
async fn constant_embedder_returns_same_vec() {
    let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
    let a = embedder.embed("anything").await.unwrap();
    let b = embedder.embed("something else").await.unwrap();
    assert_eq!(a, Some(vec![0.1, 0.2, 0.3]));
    assert_eq!(a, b, "identical regardless of input");
}

#[tokio::test]
async fn scripted_extractor_returns_in_order_then_empty() {
    let extractor = ScriptedExtractor::new(vec![
        Ok(vec!["first fact".to_string()]),
        Ok(vec!["second fact".to_string()]),
    ]);

    let r1 = extractor.extract_facts("content", &[]).await.unwrap();
    let r2 = extractor.extract_facts("content", &[]).await.unwrap();
    let r3 = extractor.extract_facts("content", &[]).await.unwrap();

    assert_eq!(r1, vec!["first fact".to_string()]);
    assert_eq!(r2, vec!["second fact".to_string()]);
    assert!(r3.is_empty(), "exhausted script yields empty, not error");
    assert_eq!(extractor.call_count(), 3);
}

#[test]
fn fixed_clock_constant() {
    let clock = FixedClock(ts());
    assert_eq!(clock.now(), ts());
    assert_eq!(clock.now(), ts(), "always the same instant");
}

#[tokio::test]
async fn scripted_reranker_returns_in_order_then_empty() {
    let docs = ["alpha".to_string(), "beta".to_string()];
    let r0 = vec![RerankResult {
        index: 0,
        score: 0.9,
        document: "alpha".into(),
    }];
    let r1 = vec![RerankResult {
        index: 1,
        score: 0.8,
        document: "beta".into(),
    }];
    let reranker = ScriptedReranker::new(vec![Ok(r0.clone()), Ok(r1.clone())]);

    let out1 = reranker.rerank("q", &docs, 2).await.unwrap();
    let out2 = reranker.rerank("q", &docs, 2).await.unwrap();
    let out3 = reranker.rerank("q", &docs, 2).await.unwrap();

    assert_eq!(out1, r0);
    assert_eq!(out2, r1);
    assert!(
        out3.is_empty(),
        "exhausted script yields empty Ok, not error"
    );
}

#[tokio::test]
async fn scripted_reranker_records_query_docs_and_top_k() {
    let docs = ["d0".to_string(), "d1".to_string(), "d2".to_string()];
    let reranker = ScriptedReranker::matching(|_q, docs, top_k| {
        Ok((0..top_k.min(docs.len()))
            .map(|i| RerankResult {
                index: i,
                score: 1.0 - i as f32 * 0.1,
                document: String::new(),
            })
            .collect())
    });

    reranker.rerank("query one", &docs, 2).await.unwrap();
    reranker.rerank("query two", &docs, 3).await.unwrap();

    let calls = reranker.calls();
    assert_eq!(
        calls,
        vec![
            ("query one".to_string(), 3, 2),
            ("query two".to_string(), 3, 3),
        ],
        "each call records (query, document_count, top_k)"
    );
}

#[tokio::test]
async fn scripted_reranker_matching_honours_top_k() {
    let docs = (0..5).map(|i| format!("d{i}")).collect::<Vec<_>>();
    let reranker = ScriptedReranker::matching(|_q, docs, top_k| {
        Ok(docs
            .iter()
            .enumerate()
            .take(top_k)
            .map(|(i, d)| RerankResult {
                index: i,
                score: 1.0,
                document: d.clone(),
            })
            .collect())
    });

    let out = reranker.rerank("q", &docs, 2).await.unwrap();
    assert_eq!(out.len(), 2, "match-mode resolver receives top_k verbatim");
    assert_eq!(out[0].index, 0);
    assert_eq!(out[1].index, 1);
}
