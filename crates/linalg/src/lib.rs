#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod eigen;
pub mod lapack;
pub mod lu;
pub mod matrix;
mod vector;

pub use eigen::eigen;
pub use vector::Vector;
