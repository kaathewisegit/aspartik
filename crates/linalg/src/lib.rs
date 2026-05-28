#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod beast_eigen;
pub mod const_matrix;
mod dim;
mod ext;
#[cfg(feature = "lapack")]
pub mod lapack;
mod matrix;

pub use const_matrix::ConstMatrixRef;
pub use dim::Dim;
pub use ext::{MatrixArrayExt, MatrixSliceExt};
pub use matrix::{MatrixMut, MatrixRef};

pub fn mul<T>(
	lhs: MatrixRef<'_, T>,
	rhs: MatrixRef<'_, T>,
	mut dst: MatrixMut<'_, T>,
) where
	T: Copy + num_traits::Num,
{
	assert_eq!(lhs.num_cols(), rhs.num_rows());
	assert_eq!(dst.num_rows(), lhs.num_rows());
	assert_eq!(dst.num_cols(), rhs.num_cols());

	let rows = dst.num_rows();
	let cols = dst.num_cols();
	let inner = lhs.num_cols();

	for i in 0..rows {
		for j in 0..cols {
			let mut sum = T::zero();
			for k in 0..inner {
				sum = sum + lhs[(i, k)] * rhs[(k, j)];
			}
			dst[(i, j)] = sum;
		}
	}
}
