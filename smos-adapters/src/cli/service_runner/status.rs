//! SCM `SetServiceStatus` wrappers and exit-code mapping.
//!
//! [`set_status`] is the single point that decides which controls a
//! state accepts: `StartPending`, `StopPending`, and `Running` all keep
//! `STOP` accepted so the operator can cancel a slow start / drain via
//! `sc stop` without SCM hanging. [`exit_code_for_result`] maps the
//! `run_server_with_shutdown` outcome onto the [`ServiceExitCode`] SCM
//! consults (together with the `failureflag` set at install time) to
//! decide whether the configured restart backoff fires.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use windows_service::service::{
    ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::ServiceStatusHandle;

/// SCM status reported while the service is live. `STOP` is always
/// accepted so `sc stop` works during start-up and drain; `Running`
/// additionally accepts `SHUTDOWN` / `PRESHUTDOWN` so the OS reboot and
/// the early pre-shutdown notification reach the control handler.
fn controls_for(state: ServiceState) -> ServiceControlAccept {
    match state {
        ServiceState::Running => {
            ServiceControlAccept::STOP
                | ServiceControlAccept::SHUTDOWN
                | ServiceControlAccept::PRESHUTDOWN
        }
        ServiceState::StartPending | ServiceState::StopPending => ServiceControlAccept::STOP,
        _ => ServiceControlAccept::empty(),
    }
}

/// Update the service status. `wait_hint` is only meaningful for the
/// `*Pending` states; SCM ignores it for `Running` / `Stopped`.
pub(super) fn set_status(
    handle: &Arc<ServiceStatusHandle>,
    state: ServiceState,
    wait_hint: Duration,
) -> Result<()> {
    let status = ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: state,
        controls_accepted: controls_for(state),
        exit_code: ServiceExitCode::Win32(0),
        checkpoint: 0,
        wait_hint,
        process_id: None,
    };
    handle
        .set_service_status(status)
        .with_context(|| format!("failed to report service status {state:?}"))
}

/// Report the terminal `STOPPED` status, reusing the existing handle
/// rather than re-registering with SCM. The handle is already wired up
/// by [`super::register_control_handler`]; re-registering at shutdown
/// risks a no-op failure (SCM tearing the process down) that would leave
/// the service without a final status and force a hard kill.
pub(super) fn set_stopped(
    handle: &Arc<ServiceStatusHandle>,
    exit_code: ServiceExitCode,
) -> Result<()> {
    let status = ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: ServiceState::Stopped,
        controls_accepted: ServiceControlAccept::empty(),
        exit_code,
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    };
    handle
        .set_service_status(status)
        .context("set_service_status(STOPPED) failed")
}

/// Map the proxy body's outcome onto the exit code SCM reads on
/// `STOPPED`. `Win32(0)` is a clean stop (no restart); any non-zero code
/// triggers the restart backoff — but ONLY when `failureflag` is enabled
/// (set at install time via `sc failureflag smos 1`), otherwise SCM
/// treats a graceful non-zero stop as success.
pub(super) fn exit_code_for_result(result: &Result<()>) -> ServiceExitCode {
    match result {
        Ok(()) => ServiceExitCode::Win32(0),
        Err(_) => ServiceExitCode::ServiceSpecific(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_code_for_result_ok_is_clean_win32_zero() {
        let ok: Result<()> = Ok(());
        assert!(matches!(
            exit_code_for_result(&ok),
            ServiceExitCode::Win32(0)
        ));
    }

    #[test]
    fn exit_code_for_result_err_is_service_specific_one() {
        let err: Result<()> = Err(anyhow::anyhow!("boom"));
        assert!(matches!(
            exit_code_for_result(&err),
            ServiceExitCode::ServiceSpecific(1)
        ));
    }

    #[test]
    fn controls_for_running_accepts_stop_shutdown_preshutdown() {
        let c = controls_for(ServiceState::Running);
        assert!(c.contains(ServiceControlAccept::STOP));
        assert!(c.contains(ServiceControlAccept::SHUTDOWN));
        assert!(c.contains(ServiceControlAccept::PRESHUTDOWN));
    }

    #[test]
    fn controls_for_pending_states_accept_only_stop() {
        for state in [ServiceState::StartPending, ServiceState::StopPending] {
            let c = controls_for(state);
            assert!(c.contains(ServiceControlAccept::STOP));
            assert!(!c.contains(ServiceControlAccept::SHUTDOWN));
            assert!(!c.contains(ServiceControlAccept::PRESHUTDOWN));
        }
    }

    #[test]
    fn controls_for_stopped_accepts_nothing() {
        let c = controls_for(ServiceState::Stopped);
        assert_eq!(c, ServiceControlAccept::empty());
    }
}
