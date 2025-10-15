mod integrands;
mod integrators;
mod naive;
mod parametrization;
mod util;
mod vectors;
use crate::vectors::{LVec, Vec3};
use num_complex::Complex64;
use pyo3::prelude::*;

#[pyclass]
pub(crate) struct IntegrationResult {
    mean: f64,
    err: f64,
}

#[pymethods]
impl IntegrationResult {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("mean = {}, error = {}", self.mean, self.err))
    }
}
struct Logger {
    logger: Py<PyAny>,
}
/// Thin wrapper around a python logger
impl Logger {
    pub fn new(logger: Py<PyAny>) -> Self {
        Self { logger }
    }
    pub fn debug(&self, msg: String) {
        Python::attach(|py| {
            self.logger.call_method1(py, "debug", (msg,)).unwrap();
        })
    }
    pub fn info(&self, msg: String) {
        Python::attach(|py| {
            self.logger.call_method1(py, "info", (msg,)).unwrap();
        })
    }
    pub fn critical(&self, msg: String) {
        Python::attach(|py| {
            self.logger.call_method1(py, "critical", (msg,)).unwrap();
        })
    }
}
/// Python wrapper for triangle integration functionality.
///
/// This struct wraps any implementation of `TriangleTrait` and exposes it to Python.
/// The `inner` field stores a boxed trait object (`Box<dyn TriangleTrait>`), which allows
/// **dynamic dispatch**: Python can call methods on `Triangle` without knowing the concrete Rust type.
//
/// Dynamic dispatch is required here because PyO3 cannot work with Rust generics.
/// Rust will call the appropriate method through the trait object at runtime.
#[pyclass]
struct Triangle {
    inner: Box<dyn TriangleTrait>,
}

#[pymethods]
impl Triangle {
    /// Performs the integration over the triangle.
    #[pyo3(signature = ())]
    fn integrate(&self) -> IntegrationResult {
        self.inner.integrate()
    }

    /// Evaluates the integrand at a given momentum vector `k`.
    fn evaluate(&self, k: Vec3) -> Complex64 {
        self.inner.evaluate(k)
    }

    /// Evaluates the integrand using a parameterized coordinate `xs`.
    fn evaluate_parameterized(&self, xs: Vec3) -> Complex64 {
        self.inner.evaluate_parameterized(xs)
    }
}
/// Trait defining the core triangle functionality.
///
/// All concrete triangle implementations must implement this trait.
/// It is `Sync + Send` so that it can be safely used in multithreaded contexts.
///
/// This trait provides a uniform interface for integration and evaluation, allowing
/// the Python wrapper to work with any Rust triangle implementation dynamically.
trait TriangleTrait: Sync + Send {
    /// Performs the integration over the triangle.
    fn integrate(&self) -> IntegrationResult;

    /// Evaluates the integrand at a given momentum vector `k`.
    fn evaluate(&self, k: Vec3) -> Complex64;

    /// Evaluates the integrand using a parameterized coordinate `xs`.
    fn evaluate_parameterized(&self, xs: Vec3) -> Complex64;
}

#[pymodule]
fn triangler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::naive::new_naive;
    use pyo3::prelude::*;
    m.add_function(wrap_pyfunction!(new_naive, m)?)?;
    m.add_class::<Triangle>()?;
    m.add_class::<Vec3>()?;
    m.add_class::<LVec>()?;
    Ok(())
}
