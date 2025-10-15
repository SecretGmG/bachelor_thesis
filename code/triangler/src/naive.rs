use crate::{
    cartesian_match, integrands::{ImprovedLTD, Integrand, NaiveIntegrand}, integrators::{Integrator, MonteCarlo, Vegas, VegasMulti}, parametrization::{Cartesian, Parametrization, Spherical}, vectors::{LVec, Vec3}, IntegrationResult, Logger, Triangle, TriangleTrait
};
use ::num_complex::Complex64;
use pyo3::{exceptions::PyValueError, *};

struct NaiveTriangle<F, I, P> where F: Integrand, I: Integrator, P : Parametrization{
    integrand: F,
    integrator: I,
    parametrization: P,
    #[allow(unused)]
    p: LVec,
    #[allow(unused)]
    q: LVec,
    #[allow(unused)]
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
    ) -> PyResult<Triangle> {
        let p = p.clone();
        let q = q.clone();
        let logger = Logger::new(logger);
        macro_rules! create_naive_triangle (
            ($integrand_:tt, $integrator_:tt, $parametrization_:tt,) => {{
            let inner = NaiveTriangle {
                integrand: $integrand_?,
                integrator: $integrator_?,
                parametrization: $parametrization_?,
                p,
                q,
                m_psi,
                logger,
            };
            PyResult::Ok(Triangle {
                inner: Box::new(inner)
            })
        }});
        cartesian_match!(
            create_naive_triangle,
            match (integrand) {
                "naive" => {
                    PyResult::Ok(NaiveIntegrand::new(p, q, m_psi))
                },
                "improved" =>{
                    PyResult::Ok(ImprovedLTD::new(p, q, m_psi))
                },
                invalid => {
                    PyResult::<NaiveIntegrand>::Err(PyValueError::new_err(format!("{invalid} is an invalid argument for integrand")))
                },
            },
            match (integrator) {
                "monte_carlo" => {
                    PyResult::Ok(MonteCarlo {})
                },
                "vegas" => {
                    PyResult::Ok(Vegas::new(20, 100_000))
                },
                "vegas_multi" => {
                    PyResult::Ok(VegasMulti::new(20, 100_000, 10))
                },
                invalid => {
                    PyResult::<MonteCarlo>::Err(PyValueError::new_err(format!("{invalid} is an invalid argument for integrator")))
                },
            },
            match (parametrization) {
                "cartesian" => {
                    PyResult::Ok(Cartesian {})
                },
                "spherical" => {
                    PyResult::Ok(Spherical {})
                },
                invalid => {
                    PyResult::<Cartesian>::Err(PyValueError::new_err(format!("{invalid} is an invalid argument for parametrization")))
                },
            },
        )
    }
