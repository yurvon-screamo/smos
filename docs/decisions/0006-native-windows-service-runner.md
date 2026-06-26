# ADR-0006: Native Windows service runner via `windows-service`

- **Status:** Accepted
- **Date:** 2026-06-26
- **Supersedes:** —

## Context

`smos service install` on Windows failed with `sc start failed:` and SCM
error **1053 (`ERROR_SERVICE_REQUEST_TIMEOUT`)**. Root cause (confirmed
via research + the diagnostic `scripts/diag-service-install.ps1`): the
`smos.exe` binary is a plain console process — it never calls
`StartServiceCtrlDispatcher`, never registers `ServiceMain`, and never
reports `SERVICE_RUNNING` to the SCM. On Linux / macOS the same install
path works because systemd / launchd supervise ANY process; the Windows
SCM requires a cooperating partner INSIDE the process. Three earlier
commits (`3ccb717`, `1c93caf`) hardened `binPath` quoting and error
surfacing, but that was red-herring work — even a perfectly-quoted
`binPath` cannot make a non-service process answer SCM.

The decision was taken to implement a **native** Windows service via the
`windows-service` crate (the standard Rust binding, used by mullvad /
nym-vpn / etc.) rather than reach for an external wrapper (NSSM / WinSW)
or fall back to Task Scheduler. The native path keeps the Windows story
symmetric with Linux systemd and macOS launchd, makes the
already-coded `sc failure` recovery actions actually fire, and adds zero
extra native dependencies beyond the Rust crate.

## Decision

### Dual-mode via a hidden `--run-as-service` flag

`format_bin_path` now emits `"<exe>" --run-as-service --config "<cfg>"`.
`bin/smos.rs::main` is a **synchronous** `fn main` that checks
`args[1] == "--run-as-service"` BEFORE building any tokio runtime and
hands control to `service_runner::run_as_service`. `service_dispatcher::start`
blocks the main thread for the whole service lifetime, so the runtime
has to be built INSIDE `ServiceMain` on the SCM worker thread — a
runtime already living on the main thread (e.g. from `#[tokio::main]`)
would collide with that nested runtime. The CLI path builds its own
runtime in `main` and drives `run_cli` (the previous async body).

### Reusable shutdown trigger

`run_server` is split into a CLI wrapper (`run_server`) and a shared body
`run_server_with_shutdown(config, CancellationToken, on_ready)`. The CLI
wires Ctrl+C / SIGTERM into the token; the service wires the SCM `Stop`
control into the SAME token. The §12 drain ordering (HTTP → extraction →
watcher) is therefore byte-identical across launch modes.

### `RUNNING` only after the listener is bound

`run_server_with_shutdown` accepts an `on_ready: Option<Box<dyn FnOnce() + Send>>`
fired once after `TcpListener::bind` succeeds. The service mode passes a
closure that flips the SCM status to `SERVICE_RUNNING` at that exact
moment, so SCM (and any `DependOnService` consumers) do not see "started"
while the server is still mid-init (model load, migrations, llama
auto-launch).

### `STOP_PENDING` the instant Stop arrives

A background tokio task watches the shutdown token and reports
`STOP_PENDING` (with `wait_hint = 45s`) the moment it fires — in parallel
with the drain triggered by the same token. This honours SCM's contract
that a service must promptly leave `RUNNING` on Stop, otherwise SCM
applies its own default grace and then `TerminateProcess`, which would
kill the §12 drain mid-flight and lose pending facts. The control
handler itself only cancels the token (it cannot report status — there
is no status handle at `register` time).

### Restart-on-error via `sc failureflag`

`set_failure_recovery` (which configures `sc failure ... actions=restart/...`)
is now paired with `set_failure_flag` (`sc failureflag smos 1`) at install
time. Without the flag, SCM treats a graceful `SERVICE_STOPPED` with a
non-zero exit code as a clean stop and never fires the configured
recovery actions. `ServiceExitCode::ServiceSpecific(1)` on a startup
error + the flag = the documented restart backoff finally works.

## Consequences / compromises

- **`Arc<ServiceStatusHandle>`** — `ServiceStatusHandle` is not `Clone`
  in `windows-service` 0.8, so the handle is shared via `Arc`. The stop
  watchdog, the `on_ready` closure, and the terminal `set_stopped` all
  clone the `Arc` cheaply and reuse the SAME registered handle rather
  than re-registering at shutdown.
- **Synchronous tracing appender** — `init_tracing_for_service` uses
  `tracing_appender::rolling` directly (NOT `non_blocking`). A service
  has no throughput pressure, and a non-blocking worker would drop the
  last buffered lines on process exit — including the terminal `error!`
  that explains why the service failed to start.
- **`controls_accepted = STOP` on `StartPending` / `StopPending`** —
  lets the operator cancel a slow start or drain via `sc stop` instead
  of SCM hanging on a non-accepted control.
- **Session 0 / LocalSystem gotcha** — a service runs as `LocalSystem`,
  whose profile is `C:\Windows\System32\config\systemprofile`, NOT the
  operator's. `print_install_summary` hard-codes that path and points
  the operator at `sc config smos env=SMOS_HOME=<path>` to redirect.
  Operators are also expected to run `smos init` first so the 643 MB
  DeBERTa / GGUF model files exist on disk before the service starts
  (Session 0 has no interactive download path).
- **No compile-time GPU features** — unchanged. The `windows-service`
  addition is windows-only and orthogonal to the runtime ORT device
  selection.

## Verification

`cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo fmt --all --check`, and `cargo t` (~935 tests, 0
failed, 6 ignored — DeBERTa) all pass. Unit tests cover
`extract_config_from_args`, `exit_code_for_result`,
`controls_for(state)`, and the `format_bin_path` / `quote_for_argv`
shapes (the latter round-trips through a full `CommandLineToArgvW`
re-implementation). End-to-end SCM lifecycle verification requires an
elevated Windows host and is left to manual / `#[ignore]` integration
testing.

## References

- `smos-adapters/src/cli/service_runner/{mod,scm,status}.rs`
- `smos-adapters/src/cli/server_runner.rs` — `run_server_with_shutdown`
- `smos-adapters/src/bin/smos.rs` — synchronous `main` + service branch
- `scripts/diag-service-install.ps1` — root-cause diagnostic
- `windows-service` crate: https://github.com/mullvad/windows-service-rs
