use crate::{RowMatrix, Vector, lapack};

impl<const N: usize> RowMatrix<f64, N, N> {
	/// Returns eigenvalues and matrices whose rows are right and left
	/// eigenvectors, in that order.
	pub fn eigen(
		&self,
	) -> (Vector<f64, N>, RowMatrix<f64, N, N>, RowMatrix<f64, N, N>) {
		eigen(self, true, true)
	}

	pub fn eigenvectors(&self) -> RowMatrix<f64, N, N> {
		let (_, right, _) = eigen(self, true, false);
		right
	}

	pub fn left_eigenvectors(&self) -> RowMatrix<f64, N, N> {
		let (_, _, left) = eigen(self, false, true);
		left
	}

	pub fn eigenvalues(&self) -> Vector<f64, N> {
		let (values, _, _) = eigen(self, false, false);
		values
	}

	pub fn inverse(&self) -> Self {
		let (lu, ipiv) = lapack::dgetrf(self);
		lapack::dgetri(&lu, &ipiv)
	}
}

fn eigen<const N: usize>(
	matrix: &RowMatrix<f64, N, N>,
	left: bool,
	right: bool,
) -> (Vector<f64, N>, RowMatrix<f64, N, N>, RowMatrix<f64, N, N>) {
	if matrix.is_symmetric() {
		let (values, vectors) = lapack::dsyev(matrix, left || right);
		(values, vectors, vectors)
	} else {
		lapack::dgeev(matrix, left, right)
	}
}
