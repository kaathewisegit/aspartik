#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod const_matrix;
mod dim;
mod eigen;
#[cfg(feature = "lapack")]
pub mod lapack;
pub mod lu;
mod matrix;

pub use const_matrix::ConstMatrixRef;
pub use dim::Dim;
pub use eigen::eigen;
pub use matrix::{MatrixMut, MatrixRef, mul};
