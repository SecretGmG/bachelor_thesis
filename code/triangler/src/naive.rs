use crate::{
    cartesian_match, integrands::{ImprovedLTD, Integrand, NaiveIntegrand}, integrators::{Integrator, MonteCarlo, Vegas}, parametrization::{Cartesian, Parametrization, Spherical}, vectors::{LVec, Vec3}, IntegrationResult, Logger, Triangle, TriangleTrait
};
use ::num_complex::Complex64;
use pyo3::*;

struct NaiveTriangle<F, I, P> where F: Integrand, I: Integrator, P : Parametrization{
    integrand: F,
    integrator: I,
    parametrization: P,
    p: LVec,
    q: LVec,
    m_psi: f64,
    logger : Logger
}

impl<F, I, P> TriangleTrait for NaiveTriangle<F, I, P>
where
    F: Integrand,
    I: Integrator,
    P: Parametrization,
{
    fn integrate(&self) -> IntegrationResult {
        self.integrator.integrate(&self.integrand, &self.parametrization, &self.logger)
    }

    fn evaluate(&self, k: Vec3) -> Complex64 {
        Complex64::new(self.integrand.evaluate(k),0.0)
    }

    fn evaluate_parameterized(&self, xs: Vec3) -> Complex64 {
        self.evaluate(self.parametrization.transform(xs).0)
    }
}
#[pyfunction]
pub fn new_naive(
        p: &LVec,
        q: &LVec,
        m_psi: f64,
        integrator: &str,
        integrand: &str,
        parametrization: &str,
        logger: Py<PyAny>
    ) -> Triangle {
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
                logger: Logger::new(logger),
            };
            Triangle {
                inner: Box::new(inner)
            }
        }});
        cartesian_match!(
            create_naive_triangle,
            match (integrand) {
                "naive" => {
                    NaiveIntegrand::new(p, q, m_psi)
                },
                "improved" =>{
                    ImprovedLTD::new(p, q, m_psi)
                },
                _ => {
                    NaiveIntegrand::new(LVec::zero(), LVec::zero(), 0.0)
                },
            },
            match (integrator) {
                "monte_carlo" => {
                    MonteCarlo {}
                },
                "vegas" => {
                    Vegas::new(5, 100_000)
                },
                _ => {
                    MonteCarlo {}
                },
            },
            match (parametrization) {
                "cartesian" => {
                    Cartesian {}
                },
                "spherical" => {
                    Spherical {}
                },
                _ => {
                    Cartesian {}
                },
            },
        )
    }
