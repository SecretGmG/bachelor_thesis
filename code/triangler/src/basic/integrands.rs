use std::f64::consts::PI;

use pyo3::{PyErr, PyResult};

use crate::vectors::{LVec, Vec3};

enum IntegrandMode {
    BasicLTD,
    ImprovedLTD,
}
fn energy(k: Vec3, qi: LVec, m_psi: f64) -> f64 {
    let spatial = qi.spatial();
    let kq = k + spatial;
    (kq.dot(&kq) + m_psi * m_psi).sqrt()
}
pub struct Integrand {
    p: LVec,
    q: LVec,
    m_psi: f64,
    mode: IntegrandMode,
}
impl Integrand {
    pub fn evaluate(&self, k: Vec3) -> f64 {
        // Define the q vectors
        let q = [LVec::new(0.0, 0.0, 0.0, 0.0), -self.q, self.p];

        // Compute energies E_i
        let e: Vec<f64> = q.iter().map(|qi| energy(k, *qi, self.m_psi)).collect();
        match self.mode {
            IntegrandMode::BasicLTD => {
                let term = |i: usize, j: usize, k: usize| {
                    let ei = e[i];
                    let ej = e[j];
                    let ek = e[k];
                    let qi_t = q[i].t();
                    let qj_t = q[j].t();
                    let qk_t = q[k].t();

                    1.0 / (2.0
                        * ei
                        * (ei + ej + (qi_t - qj_t))
                        * (ei - ej + (qi_t - qj_t))
                        * (ei + ek + (qi_t - qk_t))
                        * (ei - ek + (qi_t - qk_t)))
                };

                let sum = term(0, 1, 2) + term(1, 0, 2) + term(2, 0, 1);

                (1.0 / (2.0 * PI).powi(3)) * sum
            }
            IntegrandMode::ImprovedLTD => {
                let mut eta = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        eta[i][j] = e[i] + e[j];
                    }
                }
                // Compute sum of terms
                let sum =
                      1.0 / ((eta[0][2] - q[0].t() + q[2].t()) * (eta[1][2] - q[1].t() + q[2].t()))
                    + 1.0 / ((eta[0][1] + q[0].t() - q[1].t()) * (eta[1][2] - q[1].t() + q[2].t()))
                    + 1.0 / ((eta[0][1] + q[0].t() - q[1].t()) * (eta[0][2] + q[0].t() - q[2].t()))
                    + 1.0 / ((eta[0][2] + q[0].t() - q[2].t()) * (eta[1][2] + q[1].t() - q[2].t()))
                    + 1.0 / ((eta[0][1] - q[0].t() + q[1].t()) * (eta[1][2] + q[1].t() - q[2].t()))
                    + 1.0 / ((eta[0][1] - q[0].t() + q[1].t()) * (eta[0][2] - q[0].t() + q[2].t()));

                (2.0 * PI).powi(-3) / (2.0 * e[0] * 2.0 * e[1] * 2.0 * e[2]) * sum
            }
        }
    }
    pub fn from_args(
        mode: &str,
        p: LVec,
        q: LVec,
        m_psi: f64,
    ) -> PyResult<Self> {
        let integrand_mode = match mode {
            "basic" => IntegrandMode::BasicLTD,
            "improved" => IntegrandMode::ImprovedLTD,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "{other} is not a valid Integrand mode. Use 'basic' or 'improved'."
                )));
            }
        };

        Ok(Self {
            p,
            q,
            m_psi,
            mode: integrand_mode,
        })
    }
}