use num_complex::Complex64;

use crate::vectors::Vector3;

pub(crate) trait Integrand : Sync+Send{
    fn evaluate(&self, k: Vector3) -> Complex64;
}

pub struct NaiveIntegrand{}
impl Integrand for NaiveIntegrand {
    fn evaluate(&self, k: Vector3) -> Complex64 {
        todo!()
    }
}

pub struct ImprovedLTD{}
impl Integrand for ImprovedLTD{
    fn evaluate(&self, k: Vector3) -> Complex64 {
        todo!()
    }
}