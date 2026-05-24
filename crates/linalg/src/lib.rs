#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod const_matrix;
mod dim;
mod eigen;
#[cfg(feature = "lapack")]
pub mod lapack;
pub mod lu;
mod matrix;
mod vector;

pub use const_matrix::{ConstMatrix, ConstMatrixRef, ConstSquareMatrix};
pub use dim::Dim;
pub use eigen::eigen;
pub use matrix::{MatrixMut, MatrixRef, mul};
pub use vector::Vector;
