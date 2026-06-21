//! NLI classifier adapter.
//!
//! Single in-process implementation backs the application-layer
//! [`NliClassifier`](smos_application::ports::NliClassifier) port:
//! [`native_nli::NativeNliClassifier`] — ort + ONNX Runtime running
//! in-process against the DeBERTa-v3 ONNX export. Supports CUDA, DirectML
//! (recommended for Intel Arc on Windows), Metal and WebGPU execution
//! providers plus a CPU fallback. See [`device::DeviceKind`] for the full
//! feature matrix.
//!
//! Pure verdict aggregation (`NliResult` thresholds) lives in
//! `smos-domain::value_objects::nli`; this module only owns the IO side of
//! (premise, hypothesis) → [`NliResult`].

pub mod device;
pub mod model_cache;
pub mod native_nli;
pub mod runtime;

pub use native_nli::NativeNliClassifier;
pub use runtime::build_classifier;
