use crate::vectors::Vec3;
use std::f64::consts::PI;

pub(crate) trait Parametrization: Sync + Send {
    fn transform(&self, xs: Vec3) -> (Vec3, f64);
}

pub struct Cartesian;

impl Parametrization for Cartesian {
    fn transform(&self, xs: Vec3) -> (Vec3, f64) {
        fn poly_map(x: f64) -> f64 {
            1.0 / (1.0 - x) - 1.0 / x
        }
        fn poly_map_jac(x: f64) -> f64 {
            1.0 / (1.0 - x).powi(2) + 1.0 / x.powi(2)
        }

        let mapped = Vec3 {
            x: poly_map(xs.x),
            y: poly_map(xs.y),
            z: poly_map(xs.z),
        };

        let jac = poly_map_jac(xs.x) * poly_map_jac(xs.y) * poly_map_jac(xs.z);

        (mapped, jac)
    }
}

pub struct Spherical;

impl Parametrization for Spherical {
    fn transform(&self, xs: Vec3) -> (Vec3, f64) {
        let r = xs.x / (1.0 - xs.x);
        let r_jac = 1.0 / (1.0 - xs.x).powi(2);

        let th = xs.y * 2.0 * PI;
        let th_jac = 2.0 * PI;

        let phi = xs.z * PI;
        let phi_jac = PI;

        let (sin_phi, cos_phi) = phi.sin_cos();

        let v = Vec3 {
            x: r * sin_phi * th.cos(),
            y: r * sin_phi * th.sin(),
            z: r * cos_phi,
        };

        let jac = (r_jac * th_jac * phi_jac) * sin_phi * r.powi(2);

        (v, jac)
    }
}
