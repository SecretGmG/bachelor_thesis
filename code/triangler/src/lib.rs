mod basic;
mod parametrization;
mod vectors;
use crate::vectors::{LVec, Vec3};
use pyo3::prelude::*;
use basic::Triangle;
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
    #[allow(unused)]
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
    #[allow(unused)]
    pub fn critical(&self, msg: String) {
        Python::attach(|py| {
            self.logger.call_method1(py, "critical", (msg,)).unwrap();
        })
    }
}

#[pymodule]
fn triangler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Triangle>()?;
    m.add_class::<Vec3>()?;
    m.add_class::<LVec>()?;
    Ok(())
}