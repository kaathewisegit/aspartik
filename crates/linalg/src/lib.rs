#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod const_matrix;
mod dim;
mod eigen;
#[cfg(feature = "lapack")]
pub mod lapack;
pub mod lu;
mod matrix;

pub use const_matrix::{ConstMatrix, ConstMatrixRef, ConstSquareMatrix};
pub use dim::Dim;
pub use eigen::eigen;
pub use matrix::{MatrixMut, MatrixRef, mul};

pub fn from_diagonal<T, const N: usize>(diag: &[T; N]) -> [[T; N]; N]
where
	T: num_traits::Zero + Copy,
{
	let mut out = [[T::zero(); N]; N];

	for i in 0..N {
		out[i][i] = diag[i];
	}

	out
}
