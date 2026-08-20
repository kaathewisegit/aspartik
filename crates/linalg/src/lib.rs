#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod beast_eigen;
pub mod const_matrix;
mod dim;
mod ext;
mod matrix;

pub use const_matrix::ConstMatrixRef;
pub use dim::Dim;
pub use ext::{MatrixArrayExt, MatrixSliceExt};
pub use matrix::{MatrixMut, MatrixRef};

pub fn mul(
	lhs: MatrixRef<'_, f64>,
	rhs: MatrixRef<'_, f64>,
	mut dst: MatrixMut<'_, f64>,
) {
	assert_eq!(lhs.num_cols(), rhs.num_rows());
	assert_eq!(dst.num_rows(), lhs.num_rows());
	assert_eq!(dst.num_cols(), rhs.num_cols());

	let rows = dst.num_rows();
	let cols = dst.num_cols();
	let inner = lhs.num_cols();

	for i in 0..rows {
		for j in 0..cols {
			let mut sum = 0.0;
			for k in 0..inner {
				sum += lhs[(i, k)] * rhs[(k, j)];
			}
			dst[(i, j)] = sum;
		}
	}
}
