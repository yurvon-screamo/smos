//! NLI classifier adapter.
//!
//! Single in-process implementation backs the application-layer
//! [`NliClassifier`](smos_application::ports::NliClassifier) port:
//! [`native_nli::NativeNliClassifier`] — ort + ONNX Runtime running
//! in-process against the DeBERTa-v3 ONNX export. GPU acceleration is
//! selected at runtime: the matching ONNX Runtime DLL is downloaded on
//! first use (see [`ort_cache`]) and the ort session is built against
//! it. Supported devices: CPU, DirectML (Intel Arc / AMD / NVIDIA on
//! Windows), CUDA (NVIDIA on Windows / Linux), Metal / CoreML (Apple
//! Silicon). See [`device::DeviceKind`] for the full matrix.
//!
//! Pure verdict aggregation (`NliResult` thresholds) lives in
//! `smos-domain::value_objects::nli`; this module only owns the IO side of
//! (premise, hypothesis) → [`NliResult`].

pub mod device;
pub mod model_cache;
pub mod native_nli;
pub mod ort_cache;
pub mod runtime;

pub use native_nli::NativeNliClassifier;
pub use runtime::build_classifier;
