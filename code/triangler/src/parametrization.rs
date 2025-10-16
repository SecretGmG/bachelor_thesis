use pyo3::{exceptions::PyValueError, PyResult};

use crate::vectors::Vec3;
use std::f64::consts::PI;

pub enum Parametrization {
    Cartesian { scale: f64 },
    Spherical { scale: f64 },
}

impl Parametrization {
    pub fn transform(&self, xs: Vec3) -> (Vec3, f64) {
        match self {
            Parametrization::Cartesian { scale } => {
                // Optional polynomial mapping to extend [0,1] -> (-inf, inf)
                fn poly_map(x: f64) -> f64 {
                    1.0 / (1.0 - x) - 1.0 / x
                }
                fn poly_map_jac(x: f64) -> f64 {
                    1.0 / (1.0 - x).powi(2) + 1.0 / x.powi(2)
                }

                let mapped = Vec3 {
                    x: scale * poly_map(xs.x),
                    y: scale * poly_map(xs.y),
                    z: scale * poly_map(xs.z),
                };
                let jac = poly_map_jac(xs.x) * poly_map_jac(xs.y) * poly_map_jac(xs.z) * scale.powi(3);

                (mapped, jac)
            }
            Parametrization::Spherical { scale } => {
                let r = xs.x / (1.0 - xs.x);
                let r_jac = 1.0 / (1.0 - xs.x).powi(2);

                let th = xs.y * 2.0 * PI; // theta ∈ [0, 2π]
                let th_jac = 2.0 * PI;

                let phi = xs.z * PI; // phi ∈ [0, π]
                let phi_jac = PI;

                let (sin_phi, cos_phi) = phi.sin_cos();

                let v = Vec3 {
                    x: scale * r * sin_phi * th.cos(),
                    y: scale * r * sin_phi * th.sin(),
                    z: scale * r * cos_phi,
                };

                let jac = r_jac * r.powi(2) * th_jac * phi_jac * sin_phi * scale.powi(3);

                (v, jac)
            }
        }
    }
    pub fn from_args(mode : &str, scale : f64) -> PyResult<Self>{
        match mode {
            "cartesian" => Ok(Self::Cartesian { scale }),
            "spherical" => Ok(Self::Spherical { scale }),
            other => Err(PyValueError::new_err(format!("{other} is an invalid argument for parametrization")))
        }
    }
}
