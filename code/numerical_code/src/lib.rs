use pyo3::prelude::*;

#[inline]
fn squared(v: &Vec<f64>) -> f64 {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

#[inline]
fn add(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    return vec![v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]];
}

#[inline]
fn sub(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    return vec![v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
}

#[inline]
fn spatial(v: &Vec<f64>) -> Vec<f64> {
    return vec![v[1], v[2], v[3]];
}

#[pyfunction]
fn ltd_triangle(m_psi: f64, k: Vec<f64>, q: Vec<f64>, p: Vec<f64>) -> PyResult<f64> {
    // Implement the improved LTD expression here (Ex 2.4)
    let mut res: f64 = 0.;
    Ok(res)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn numerical_code(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ltd_triangle, m)?)?;
    Ok(())
}
