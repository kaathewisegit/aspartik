mod eigen;
mod lu;

pub use eigen::eigen;
pub use lu::{LU, LuError, inverse};

use num_traits::Num;

use crate::{MatrixMut, MatrixRef};

pub fn mul<T>(
	lhs: MatrixRef<'_, T>,
	rhs: MatrixRef<'_, T>,
	mut dst: MatrixMut<'_, T>,
) where
	T: Copy + Num,
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
