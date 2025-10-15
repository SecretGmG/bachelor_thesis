use pyo3::prelude::*;
use std::ops::{Add, Sub, Neg};

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Vec3{
    pub fn zero()->Self{
        Self::new(0.0,0.0,0.0)
    }
}

#[pymethods]
impl Vec3 {
    #[new]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn squared(&self) -> f64 {
        self.dot(self)
    }
}

// Rust operator overloading
impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct LVec {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub t: f64,
}
impl LVec{
    pub fn zero()->Self{
        Self::new(0.0,0.0,0.0,0.0)
    }
}

#[pymethods]
impl LVec {
    #[new]
    pub fn new(x: f64, y: f64, z: f64, t: f64) -> Self {
        Self { x, y, z, t }
    }

    pub fn spatial(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    // Python dunder methods
    fn __add__(&self, other: &Self) -> Self {
        *self + *other
    }

    fn __sub__(&self, other: &Self) -> Self {
        *self - *other
    }

    fn __neg__(&self) -> Self {
        -*self
    }
}

// Rust operator overloading
impl Add for LVec {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            t: self.t + rhs.t,
        }
    }
}

impl Sub for LVec {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            t: self.t - rhs.t,
        }
    }
}

impl Neg for LVec {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            t: -self.t,
        }
    }
}
