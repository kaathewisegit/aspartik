#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod const_matrix;
mod dim;
mod ext;
#[cfg(feature = "lapack")]
pub mod lapack;
pub mod math;
mod matrix;

pub use const_matrix::ConstMatrixRef;
pub use dim::Dim;
pub use ext::{MatrixArrayExt, MatrixSliceExt};
pub use matrix::{MatrixMut, MatrixRef};
