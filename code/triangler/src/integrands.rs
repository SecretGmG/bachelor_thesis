use std::f64::consts::PI;

use crate::vectors::{Vec3, LVec};

pub(crate) trait Integrand : Sync+Send{
    fn evaluate(&self, k: Vec3) -> f64;
}
pub struct NaiveIntegrand{ p: LVec, q: LVec, m_psi: f64, }


fn energy(k: Vec3,qi: LVec,m_psi:f64)->f64{
    let spatial = qi.spatial();
    let kq = k + spatial;
    (kq.dot(&kq) + m_psi * m_psi).sqrt()
}

impl Integrand for NaiveIntegrand {
    fn evaluate(&self, k: Vec3) -> f64 {
        // Define the q vectors
        let q = [
            LVec::new(0.0, 0.0, 0.0, 0.0),
            -self.q,
            self.p,
        ];

        // Compute energies E_i
        let e: Vec<f64> = q.iter()
            .map(|qi| {
                energy(k, *qi, self.m_psi)
            })
            .collect();

        // Helper closure for a single term
        let term = |i: usize, j: usize, k: usize| {
            let ei = e[i];
            let ej = e[j];
            let ek = e[k];
            let qi_t = q[i].t();
            let qj_t = q[j].t();
            let qk_t = q[k].t();

            1.0 / (
                2.0 * ei *
                (ei + ej + (qi_t - qj_t)) *
                (ei - ej + (qi_t - qj_t)) *
                (ei + ek + (qi_t - qk_t)) *
                (ei - ek + (qi_t - qk_t))
            )
        };

        let sum = term(0,1,2) + term(1,0,2) + term(2,0,1);

        (1.0 / (2.0 * PI).powi(3)) * sum
    }
}
impl NaiveIntegrand{
    pub fn new(p: LVec, q: LVec, m_psi : f64)-> Self{
        Self{p,q,m_psi}
    }
}

pub struct ImprovedLTD {
    p: LVec,
    q: LVec,
    m_psi: f64,
}

impl Integrand for ImprovedLTD {
    fn evaluate(&self, k: Vec3) -> f64 {
        // Define the q vectors
        let q = [
            LVec::new(0.0, 0.0, 0.0, 0.0),
            -self.q,
            self.p,
        ];

        // Compute energies E_i
        let e: Vec<f64> = q.iter()
            .map(|qi| {
                energy(k, *qi, self.m_psi)
            })
            .collect();
        // Compute shifts
        let q = [0.0, self.q.t(), -self.p.t()];

        // Compute eta matrix: eta[i][j] = ose[i] + ose[j]
        let mut eta = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                eta[i][j] = e[i] + e[j];
            }
        }
        // Compute sum of terms
        let sum =
            1.0 / ((eta[0][2] - q[0] + q[2]) * (eta[1][2] - q[1] + q[2])) +
            1.0 / ((eta[0][1] + q[0] - q[1]) * (eta[1][2] - q[1] + q[2])) +
            1.0 / ((eta[0][1] + q[0] - q[1]) * (eta[0][2] + q[0] - q[2])) +
            1.0 / ((eta[0][2] + q[0] - q[2]) * (eta[1][2] + q[1] - q[2])) +
            1.0 / ((eta[0][1] - q[0] + q[1]) * (eta[1][2] + q[1] - q[2])) +
            1.0 / ((eta[0][1] - q[0] + q[1]) * (eta[0][2] - q[0] + q[2]));

        (2.0 * PI).powi(-3) / (2.0 * e[0] * 2.0 * e[1] * 2.0 * e[2]) * sum
    }
}

impl ImprovedLTD {
    pub fn new(p: LVec, q: LVec, m_psi: f64) -> Self {
        Self { p, q, m_psi }
    }
}
