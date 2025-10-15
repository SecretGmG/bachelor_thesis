use crate::vectors::Vector3;

pub(crate) trait Parametrization: Sync+Send{
    fn transform(&self, xs: Vector3) -> (Vector3, f64);
}

pub struct Cartesian{}
impl Parametrization for Cartesian{
    fn transform(&self, xs: Vector3) -> (Vector3, f64) {
        todo!()
    }
}