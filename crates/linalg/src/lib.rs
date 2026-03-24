#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod lapack;
pub mod matrix;
mod vector;

pub use vector::{Vector, VectorNum};
