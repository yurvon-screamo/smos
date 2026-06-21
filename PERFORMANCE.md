# Test Performance Baseline

Last measured: 2026-06-20
Toolchain: rustc 1.96.0 (stable), Windows / MSVC toolchain
Hardware: developer laptop (CPU bound on cold compiles)

## Methodology

- `Measure-Command { cargo <cmd> 2>&1 | Out-Null }` for cold timings (after
  `cargo clean`-equivalent state on the relevant crates).
- "Warm" timings are the second invocation of the same command — Cargo only
  re-runs the test harness; no compile work.
- Test thread counts follow the alias defaults (`--test-threads=4` for
  `cargo t` / `cargo tf`, `--test-threads=2` for `cargo tall`).

## Current strategy

Every test that does NOT carry `#[ignore]` runs under the default `cargo t`.
The previous feature-gate tier system (`slow-tests` / `surrealdb-tests` /
`legacy-nli-tests` / `native-nli-tests`) was removed: the gates were code
smell that hid pre-existing regressions and made `cargo t --workspace` (the
plain command a developer types) silently skip every `e2e_*` binary. Tests
that genuinely need an external dependency carry `#[ignore = "<reason>"]`
instead and run via `cargo tall` (`--include-ignored`).

| Alias       | Resolves to                                                | Tests run          |
|-------------|------------------------------------------------------------|--------------------|
| `cargo tf`  | `smos-domain` + `smos-application` only                    | 351                |
| `cargo t`   | Workspace unit tests + every embedded-SurrealDB e2e binary | 665 (+ 5 ignored)  |
| `cargo ti`  | Alias kept for backwards compat — same scope as `cargo t` | 665                |
| `cargo tall`| Workspace + `--include-ignored` (643 MB DeBERTa + live Ollama `#[ignore]` tests) | 665 + 5 ignored |

### Top slowest test binaries

Runtimes below are from a warm `cargo t` invocation. The total warm `cargo t`
wall-clock is ~60 s; the binaries are listed in descending order of cost.

| Binary                          | Tests (passed) | Ignored | Runtime  |
|---------------------------------|----------------|---------|----------|
| `e2e_session_watcher`           | 11             | 0       | ~10 s    |
| `e2e_extraction`                | 14             | 0       | ~6 s     |
| `e2e_defensive`                 | 6              | 0       | ~3 s     |
| `surreal_store_integration`     | 22             | 0       | ~3 s     |
| `e2e_passthrough`               | 6              | 0       | ~3 s     |
| `e2e_finalize`                  | 17             | 0       | ~3 s     |
| `e2e_enrichment`                | 15             | 0       | ~2 s     |
| `e2e_import`                    | 16             | 0       | ~2 s     |
| `e2e_request`                   | 5              | 0       | ~2 s     |
| `e2e_server`                    | 3              | 0       | ~1 s     |
| `spike_surrealdb_syntax`        | 1              | 0       | ~1 s     |

`smos-domain` (200 tests, ~0.02 s), `smos-application` (141 lib tests +
17 `port_shape` integration tests, ~0.02 s), `smos-adapters` lib
(181 tests, ~0.5 s), `doctor_unit` (17 tests, ~0.00 s), and the bin
unittests (~0 s) round out the suite.

### `#[ignore]` inventory

Five tests carry `#[ignore]`:

- 5 in `native_nli_tests` — `requires 643MB DeBERTa ONNX model download`.

The previous twelve `#[ignore]`s were reduced to five by fixing the
underlying query mistakes (NOT engine bugs):

- `array::contains(...)` is not a SurrealQL function — switched to the
  `CONTAINS` operator in `list_memory_keys_for_session`.
- `array::difference(a, b)` is the symmetric difference A△B — switched
  `remove_pending_owned` and `DEDUP_AND_MARK_TX` to `array::complement`
  (the relative complement A\B they actually need).
- The two `e2e_import` ignores hid a mock-embeddings bug (the wiremock
  returned the same vector for every fact, so the persist-facts Layer 2
  semantic dedup collapsed them). Replaced with a stateful responder that
  yields distinct orthogonal unit embeddings per call.

## CPU usage notes

- `cargo t` / `cargo tf` use `--test-threads=4` — caps the parallel test
  workers at 4 instead of the cargo default (number of logical CPUs, 8-16
  on a typical laptop).
- `cargo tall` uses `--test-threads=2` — SurrealDB RocksDB startup is
  IO-heavy and the model download is bandwidth-bound; lower parallelism
  prevents disk thrash while still keeping the suite moving.
- Override per invocation with `-- --test-threads=N` or
  `RUST_TEST_THREADS=N`.

## Future work

- Share a single embedded SurrealDB instance per test binary via
  `OnceLock<RwLock<...>>` instead of one `tempdir` + RocksDB per test —
  would cut `cargo t` warm time roughly in half.
- Investigate why `e2e_session_watcher` (~10 s) is the slowest binary in
  the suite — likely the watcher's `tokio::time::sleep` polling loop.
- Consider `nextest` as a faster drop-in test runner if the alias-based
  approach ever stops being enough.
