# SMOS Refactoring Baseline (Slice R0)

> Frozen contract for every subsequent refactor slice (R1+). Each slice MUST end
> with these same numbers (or a strictly-additive superset). A regression is any
> drop in `passed`, any new `failed`, or any change to the `ignored` set.

## Environment

- Workspace: `D:\smos` (Rust 2024 edition, MSRV 1.96).
- Crates: `smos-domain`, `smos-application`, `smos-adapters`.
- Test aliases defined in `.cargo/config.toml`: `cargo t` (default, excludes
  `#[ignore]`), `cargo tf` (domain + application only), `cargo tall`
  (includes `#[ignore]`).

## Baseline numbers (R0)

| Check | Command | Result |
|---|---|---|
| Workspace tests | `cargo t` | **886 passed**, **0 failed**, **6 ignored** |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` | **0 warnings** |
| Formatting | `cargo fmt --all --check` | **clean** |

The "898 tests" figure cited in `REFACTOR_PLAN.md` counts every declared test
attribute (`#[test]` + `#[tokio::test]`). The runnable baseline under
`cargo t` is **886 passed + 6 ignored = 892**, matching the attribute count
minus 6 tests deliberately gated behind `#[ignore]`.

## Ignored tests (contract: this exact set MUST be preserved by every [BP] slice)

| # | Test | Reason | Category |
|---|---|---|---|
| 1 | `native_nli_classifies_canonical_contradiction` | requires 643MB DeBERTa ONNX model download | external dep (model) |
| 2 | `native_nli_classifies_canonical_entailment` | requires 643MB DeBERTa ONNX model download | external dep (model) |
| 3 | `native_nli_classifies_canonical_neutral` | requires 643MB DeBERTa ONNX model download | external dep (model) |
| 4 | `native_nli_handles_long_input_without_panicking` | requires 643MB DeBERTa ONNX model download | external dep (model) |
| 5 | `native_nli_softmax_distribution_sums_to_one` | requires 643MB DeBERTa ONNX model download | external dep (model) |
| 6 | `llama_server::health::tests::is_port_responding_returns_false_for_dead_port` | TOCTOU: a free port can be reassigned between the listener drop and the probe | test-quality (see Slice C6 in `REFACTOR_PLAN.md` — out of scope for Batch 1) |

Per-binary breakdown of the 886 passed (ordered as emitted by `cargo t`):

| Binary / target | passed | ignored |
|---|---:|---:|
| `smos` lib (unittests) | 375 | 1 (TOCTOU health) |
| `smos_application` lib (unittests) | 164 | 0 |
| `smos_domain` lib (unittests) | 204 | 0 |
| integration test: `e2e_session_watcher` | 22 | 0 |
| integration test: `surreal_store_integration` | 19 | 0 |
| integration test: `e2e_finalize` | 16 | 0 |
| integration test: `e2e_extraction` | 15 | 0 |
| integration test: `e2e_server` | 14 | 0 |
| integration test: `port_shape` | 14 | 0 |
| integration test: `spike_surrealdb_syntax` | 11 | 0 |
| integration test: `e2e_request` | 11 | 0 |
| integration test: `e2e_passthrough` | 6 | 0 |
| integration test: `e2e_defensive` | 6 | 0 |
| integration test: `native_nli_integration` | 0 | 5 (DeBERTa) |
| integration test: `import_runner` | 5 | 0 |
| integration test: `agent_tests` | 3 | 0 |
| integration test: `doctor_unit` | 1 | 0 |
| doc-tests (smos, smos_application, smos_domain, others) | 0 | 0 |
| **Total** | **886** | **6** |

(Binary names are inferred from emitted counts; the binding contract is the
top-line totals: 886 passed / 0 failed / 6 ignored, 0 clippy warnings, fmt
clean.)

## How to re-verify a slice

```bash
cargo t                                                    # 886 passed, 0 failed, 6 ignored
cargo clippy --workspace --all-targets -- -D warnings      # 0 warnings
cargo fmt --all --check                                    # exit 0
```

If any of these drifts for a `[BP]` slice, the slice has introduced a
regression and MUST be fixed before proceeding.
