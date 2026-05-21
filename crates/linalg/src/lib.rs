#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod const_matrix;
mod eigen;
#[cfg(feature = "lapack")]
pub mod lapack;
pub mod lu;
mod vector;

pub use const_matrix::{ConstMatrix, ConstSquareMatrix};
pub use eigen::eigen;
pub use vector::Vector;
