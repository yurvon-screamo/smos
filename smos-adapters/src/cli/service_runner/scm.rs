//! `ServiceMain` body, control handler registration, and runtime wiring.
//!
//! Everything here runs on the worker thread SCM spawns for `ServiceMain`
//! — never on the process main thread, which is blocked inside
//! [`service_dispatcher::start`] for the whole service lifetime. SCM
//! status reporting helpers live in [`status`].

use std::ffi::OsString;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use windows_service::define_windows_service;
use windows_service::service::{ServiceControl, ServiceExitCode, ServiceState};
use windows_service::service_control_handler::{
    self, ServiceControlHandlerResult, ServiceStatusHandle,
};
use windows_service::service_dispatcher;

use crate::cli::server_runner::run_server_with_shutdown;
use crate::cli::service::SERVICE_NAME;
use crate::cli::tracing_setup::init_tracing_for_service;

use super::status::{exit_code_for_result, set_status, set_stopped};

/// Hidden flag injected into the SCM `binPath` so the unified binary can
/// tell a user-driven `smos serve` apart from an SCM-driven launch.
pub const SERVICE_RUN_FLAG: &str = "--run-as-service";

/// SCM `wait_hint` for `START_PENDING`. SMOS does most heavy IO at
/// `smos init` time (GGUF + ONNX downloads), so a service start only pays
/// the in-memory ORT / tokenizers warm-up — still measured in seconds on
/// cold hardware. 120s keeps SCM's 30s `ERROR_SERVICE_REQUEST_TIMEOUT`
/// (1053) from racing a legitimate startup.
const START_PENDING_WAIT_HINT: Duration = Duration::from_secs(120);

/// SCM `wait_hint` for `STOP_PENDING`, reported the instant a stop-class
/// control arrives. Covers the §12 unified shutdown deadline plus a
/// margin for the watcher drain so SCM does not `TerminateProcess` the
/// service mid-drain (which would lose pending facts).
const STOP_PENDING_WAIT_HINT: Duration = Duration::from_secs(45);

/// Register `ServiceMain` with the SCM dispatcher and block until the
/// service stops. Returns only if the dispatcher itself fails to start
/// (the binary was launched by hand, not by SCM) — a normal service
/// lifetime ends inside `ServiceMain` via `set_service_status(STOPPED)`.
pub fn run_as_service() -> Result<()> {
    service_dispatcher::start(SERVICE_NAME, ffi_service_main)
        .with_context(|| "failed to start SCM dispatcher (is smos.exe running under SCM?)")
}

define_windows_service!(ffi_service_main, service_main);

/// `ServiceMain`: invoked by SCM on a worker thread. Owns the full
/// service lifetime — runtime build, status reporting, and the proxy
/// body. [`run_service_body`] reports the terminal `STOPPED` status
/// itself (it holds the only status handle), so this function only logs
/// on error; a pre-registration failure leaves the service in
/// `START_PENDING` and SCM applies its own timeout.
fn service_main(arguments: Vec<OsString>) {
    // Load the operator-profile env written by `smos service install`
    // BEFORE any `smos_home()` call so config resolution + tracing land
    // in the operator's profile, not LocalSystem's systemprofile. The
    // registry `Environment` value is unreliable across Windows versions
    // (SCM does not always apply it to a LocalSystem service), so the
    // install writes a plain `KEY=VALUE` file next to the binary and we
    // set_var the values here at process startup, race-free (no other
    // thread is alive yet — runtime + handler come later).
    apply_operator_env_file();

    // The config path is resolved via the SAME chain the CLI uses
    // (`--config` override > `./smos.toml` > `~/.smos/config.toml`) so
    // the service honours the operator's SMOS_HOME (loaded above) and
    // does not need a brittle absolute path baked into binPath. Without
    // this fallback the service would `SmosConfig::load("")`, hit the
    // defaults branch, and fail validation with "providers must not be
    // empty".
    let cli_override = extract_config_from_args(arguments);
    let override_opt = if cli_override.is_empty() {
        None
    } else {
        Some(cli_override.as_str())
    };
    let config_path = crate::cli::init_runner::resolve_effective_config_path(override_opt)
        .to_string_lossy()
        .into_owned();
    // catch_unwind so a panic inside run_service_body (e.g. a future
    // `.expect` in a third-party crate) is OBSERVED by the operator via
    // `smos service status` instead of aborting the process silently —
    // WIN32_EXIT_CODE 1067 with no log line is the failure mode this
    // guards against. The tracing subscriber installed by
    // init_tracing_for_service is already live, so the panic payload
    // reaches the same rolling-file log every other service line does.
    //
    // NOTE: STOPPED is NOT reported to SCM on a caught panic — the
    // status_handle lives inside run_service_body and is not in scope
    // here. SCM therefore continues to see START_PENDING until
    // START_PENDING_WAIT_HINT (120s) elapses, then TerminateProcess.
    // The improvement over the bare-panic status quo is observability
    // (the operator sees the panic message in the log); the SCM-side
    // transition stays the same. Lifting the handle into service_main
    // is a follow-up.
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_service_body(config_path)
    }));
    match panic_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::error!(error = %format!("{e:#}"), "smos service exited with error");
        }
        Err(payload) => {
            tracing::error!(
                panic = %panic_payload_string(payload),
                "smos service panicked"
            );
        }
    }
}

/// Best-effort string extraction out of a `catch_unwind` payload. Covers
/// the two idiomatic panic shapes — `panic!("literal")` (`&'static str`)
/// and `panic!("{}", x)` / `.expect("...")` (`String`); everything else
/// falls back to a fixed marker so the log line still goes out.
fn panic_payload_string(payload: Box<dyn std::any::Any + Send>) -> String {
    // `downcast` consumes the box and returns it back in the Err arm,
    // so the chain is ownership-correct: try String, on miss recover
    // the box and try `&'static str`, on miss return the fallback.
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(s) => (*s).to_string(),
            Err(_) => "<non-string panic payload>".to_string(),
        },
    }
}

/// Locate `<binary_dir>/smos-service.env`, parse it, and `set_var` every
/// pair into the process environment. Best-effort: when the file is
/// absent (older install, custom binary location without the file) the
/// service keeps running with the LocalSystem default environment — the
/// failure mode degrades to the pre-fix behaviour (systemprofile paths)
/// rather than aborting the service start.
fn apply_operator_env_file() {
    let Some(binary_dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
    else {
        return;
    };
    let Some(pairs) = crate::cli::service::env_file::load_env_file(&binary_dir) else {
        return;
    };
    // SAFETY: service_main is the first user code the SCM worker thread
    // executes — no other thread exists yet (the tokio runtime is built
    // inside run_service_body, the control handler is registered after
    // that). set_var is therefore race-free.
    unsafe { crate::cli::service::env_file::apply_env_vars(&pairs) };
}

fn run_service_body(config_path: String) -> Result<()> {
    log_nonfatal(
        init_tracing_for_service(),
        "service tracing init failed; logs will be lost",
    );

    let (status_handle, shutdown) = register_control_handler()?;
    set_status(
        &status_handle,
        ServiceState::StartPending,
        START_PENDING_WAIT_HINT,
    )?;

    let runtime = match build_runtime() {
        Ok(rt) => rt,
        Err(e) => {
            log_nonfatal(
                set_stopped(&status_handle, ServiceExitCode::ServiceSpecific(1)),
                "failed to report STOPPED after runtime build failure",
            );
            return Err(e);
        }
    };

    // Stop-class watchdog: SCM's contract requires the service to move
    // to STOP_PENDING (with a wait_hint) PROMPTLY after receiving Stop,
    // otherwise SCM applies its own default grace and then
    // TerminateProcess — killing the §12 drain mid-flight. The control
    // handler can only cancel the token (it has no status handle), so a
    // background task watches the token and flips the status the instant
    // it fires, in parallel with the drain triggered by the same token.
    spawn_stop_watchdog(&runtime, status_handle.clone(), shutdown.clone());

    // Readiness hook: SCM (and any DependOnService consumers) must not
    // see RUNNING until the HTTP listener is bound, otherwise `sc start`
    // returns success on a port that is not yet accepting connections.
    // `run_server_with_shutdown` fires this callback once the listener
    // is live, before entering the accept loop.
    //
    // The token check guards the Stop-during-init race: if the operator
    // cancels during the (potentially long) ORT warm-up, the watchdog
    // already reported STOP_PENDING and we must NOT overwrite it with
    // RUNNING — otherwise SCM observes the illegal sequence
    // StartPending → StopPending → Running → ... .
    let ready_handle = status_handle.clone();
    let ready_token = shutdown.clone();
    let on_ready = Box::new(move || {
        if ready_token.is_cancelled() {
            return;
        }
        log_nonfatal(
            set_status(&ready_handle, ServiceState::Running, Duration::default()),
            "failed to report SERVICE_RUNNING",
        );
    });

    let result = runtime.block_on(run_server_with_shutdown(
        &config_path,
        shutdown,
        Some(on_ready),
    ));

    let exit_code = exit_code_for_result(&result);
    log_nonfatal(
        set_stopped(&status_handle, exit_code),
        "failed to report final STOPPED",
    );
    result
}

/// Build the multi-threaded tokio runtime used for the proxy body. Kept
/// as a single helper so the worker-thread / naming policy has one
/// tuning point (the CLI path builds its own runtime in `bin/smos.rs`
/// for the same reason — a different thread context).
fn build_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime")
}

/// Spawn a task that translates the shutdown token's cancellation into
/// an immediate `STOP_PENDING` report. Lives only until the token fires
/// once; the runtime is dropped after the drain completes, which aborts
/// the task if Stop never arrived.
fn spawn_stop_watchdog(
    runtime: &tokio::runtime::Runtime,
    handle: Arc<ServiceStatusHandle>,
    shutdown: CancellationToken,
) {
    runtime.spawn(async move {
        shutdown.cancelled().await;
        log_nonfatal(
            set_status(&handle, ServiceState::StopPending, STOP_PENDING_WAIT_HINT),
            "failed to report STOP_PENDING on Stop",
        );
    });
}

/// Register the SCM control handler and pair it with a [`CancellationToken`]
/// that is cancelled on every stop-class control. The handler does NOT
/// report status itself — it has no status handle (chicken-and-egg with
/// `register`'s return); a background task in [`run_service_body`] watches
/// the token and flips the status. `Interrogate` returns `NoError` (SCM
/// probes liveness periodically); unsupported controls return
/// `NotImplemented` so SCM knows they are not silently dropped.
fn register_control_handler() -> Result<(Arc<ServiceStatusHandle>, CancellationToken)> {
    let shutdown = CancellationToken::new();
    let stop_token = shutdown.clone();
    let handler = move |control: ServiceControl| -> ServiceControlHandlerResult {
        match control {
            ServiceControl::Stop | ServiceControl::Shutdown | ServiceControl::Preshutdown => {
                stop_token.cancel();
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };
    let handle = service_control_handler::register(SERVICE_NAME, handler)
        .map(Arc::new)
        .context("failed to register service control handler")?;
    Ok((handle, shutdown))
}

/// Parse the `--config <path>` pair out of the arguments SCM forwarded
/// from `binPath`. Returns an empty `String` when no `--config` is
/// present so the caller can apply the standard config search chain.
pub(super) fn extract_config_from_args(args: Vec<OsString>) -> String {
    let mut iter = args.into_iter().map(|s| s.to_string_lossy().into_owned());
    while let Some(arg) = iter.next() {
        if arg == "--config"
            && let Some(value) = iter.next()
        {
            return value;
        }
    }
    String::new()
}

fn log_nonfatal(result: Result<()>, context: &str) {
    if let Err(e) = result {
        tracing::warn!(error = %format!("{e:#}"), "{context}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn extract_config_from_args_finds_config_value() {
        let args = os(&["--run-as-service", "--config", "C:\\smos\\smos.toml"]);
        assert_eq!(extract_config_from_args(args), "C:\\smos\\smos.toml");
    }

    #[test]
    fn extract_config_from_args_ignores_unrelated_flags() {
        let args = os(&["--run-as-service", "--foo", "bar", "--config", "X.toml"]);
        assert_eq!(extract_config_from_args(args), "X.toml");
    }

    #[test]
    fn extract_config_from_args_returns_empty_when_missing() {
        let args = os(&["--run-as-service"]);
        assert_eq!(extract_config_from_args(args), "");
    }

    #[test]
    fn extract_config_from_args_returns_empty_when_value_missing() {
        let args = os(&["--run-as-service", "--config"]);
        assert_eq!(extract_config_from_args(args), "");
    }

    #[test]
    fn service_run_flag_is_stable() {
        assert_eq!(SERVICE_RUN_FLAG, "--run-as-service");
    }

    #[test]
    fn panic_payload_string_extracts_string_panic() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("explode!".to_string());
        assert_eq!(panic_payload_string(payload), "explode!");
    }

    #[test]
    fn panic_payload_string_extracts_static_str_panic() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("static literal");
        assert_eq!(panic_payload_string(payload), "static literal");
    }

    #[test]
    fn panic_payload_string_falls_back_for_non_string_payload() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42_i32);
        assert_eq!(panic_payload_string(payload), "<non-string panic payload>");
    }
}
