use pyo3::prelude::*;
#[pyclass]
#[derive(Clone)]
pub struct Vector3{
    x : f64,
    y : f64,
    z : f64
}
#[pyclass]
#[derive(Clone)]
pub struct Vector4{
    spacial : Vector3,
    t : f64
}