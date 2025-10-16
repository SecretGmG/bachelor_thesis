mod integrands;
mod integrators;

use crate::{
    basic::{integrands::Integrand, integrators::Integrator}, parametrization::Parametrization, vectors::{LVec, Vec3}, IntegrationResult, Logger
};
use pyo3::prelude::*;

/// Python wrapper for triangle integration functionality.
///
/// This struct wraps any implementation of `TriangleTrait` and exposes it to Python.
/// The `inner` field stores a boxed trait object (`Box<dyn TriangleTrait>`), which allows
/// **dynamic dispatch**: Python can call methods on `Triangle` without knowing the concrete Rust type.
//
/// Dynamic dispatch is required here because PyO3 cannot work with Rust generics.
/// Rust will call the appropriate method through the trait object at runtime.
#[pyclass]
pub struct Triangle {
    integrand: Integrand,
    integrator: Integrator,
    parametrization: Parametrization,
    logger: Logger,
}

#[pymethods]
impl Triangle {
    /// Performs the integration over the triangle.
    #[pyo3(signature = ())]
    fn integrate(&self) -> IntegrationResult {
        self.integrator
            .integrate(&self.integrand, &self.parametrization, &self.logger)
    }

    /// Evaluates the integrand at a given momentum vector `k`.
    fn evaluate(&self, k: Vec3) -> f64 {
        self.integrand.evaluate(k)
    }

    /// Evaluates the integrand using a parameterized coordinate `xs`.
    fn evaluate_parameterized(&self, xs: Vec3) -> f64 {
        self.evaluate(self.parametrization.transform(xs).0)
    }

    #[new]
    fn new(
        p: &LVec,
        q: &LVec,
        m_psi: f64,
        integrator: &str,
        integrand: &str,
        parametrization: &str,
        logger: Py<PyAny>,
    ) -> PyResult<Triangle> {
        let p = p.clone();
        let q = q.clone();
        let logger = Logger::new(logger);
        Ok(Self {
            integrand: Integrand::from_args(integrand,p, q, m_psi)?,
            integrator: Integrator::from_args(integrator, 10, 100_000, 10)?,
            parametrization: Parametrization::from_args(parametrization, 1.0)?,
            logger,
        })
    }
}
