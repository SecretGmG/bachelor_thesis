mod util;
mod parametrization;
mod vectors;
mod integrands;
mod integrators;

use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::PyErr;

use crate::{integrands::{Integrand, NaiveIntegrand}, integrators::{IntegrationResult, Integrator, MonteCarlo}, parametrization::{Cartesian, Parametrization}, vectors::{Vector3, Vector4}};

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
    pub fn new_naive(
        &self,
        p: &Vector4,
        q: &Vector4,
        m_psi: f64,
        integrator: &str,
        integrand: &str,
        parametrization: &str,
    ) -> PyResult<Self> {
     let p = p.clone();
        let q = q.clone();
        macro_rules! create_naive_triangle (
            ($integrand_:tt, $integrator_:tt, $parametrization_:tt,) => {{
            let inner = NaiveTriangle {
                integrand: $integrand_,
                integrator: $integrator_,
                parametrization: $parametrization_,
                p,
                q,
                m_psi,
            };
            Ok(Self {
                inner: Box::new(inner),
            })
        }});
        cartesian_match!(
            create_naive_triangle,
            match (integrand) {
                "naive" => {NaiveIntegrand{}},
                _ => {NaiveIntegrand{}},
            },
            match (integrator) {
                "monte_carlo" => {MonteCarlo{}},
                _ => {MonteCarlo{}},
            },
            match (parametrization) {
                "cartesian" => {Cartesian{}},
                _ => {Cartesian{}},
            },
        )
    }
}
struct NaiveTriangle<F,I,P>{
    integrand:F,
    integrator:I,
    parametrization:P,
    p:Vector4,
    q:Vector4,
    m_psi:f64,
}

impl<F,I,P> TriangleTrait for NaiveTriangle<F,I,P>
where 
F: Integrand,
I: Integrator,
P: Parametrization
{
    fn integrate(&self) -> IntegrationResult {
        self.integrator.integrate()
    }

    fn evaluate(&self, k: Vector3) -> Complex64 {
        self.integrand.evaluate(k)
    }

    fn evaluate_parameterized(&self, xs: Vector3) -> Complex64 {
        self.evaluate(self.parametrization.transform(xs).0)
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
    fn evaluate(&self, k: Vector3) -> Complex64;

    /// Evaluates the integrand using a parameterized coordinate `xs`.
    fn evaluate_parameterized(&self, xs: Vector3) -> Complex64;
}

