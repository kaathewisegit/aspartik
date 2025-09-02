use nalgebra::{Const, Dim, Matrix, OMatrix, OVector, Scalar, Storage};
use num_traits::Num;

use crate::{RowMatrix, Vector};

impl<T: Copy + Scalar, const N: usize, const M: usize> From<&RowMatrix<T, N, M>>
	for OMatrix<T, Const<N>, Const<M>>
{
	fn from(value: &RowMatrix<T, N, M>) -> Self {
		OMatrix::from_row_slice_generic(Const, Const, value.as_slice())
	}
}

impl<
	T: Copy + Scalar + Num,
	const N: usize,
	const M: usize,
	D: Dim,
	S: Storage<T, D, D>,
> From<Matrix<T, D, D, S>> for RowMatrix<T, N, M>
{
	fn from(value: Matrix<T, D, D, S>) -> Self {
		assert_eq!(value.shape(), (N, M));
		let mut out = Self::zeros();

		for i in 0..N {
			for j in 0..M {
				out[i][j] = value[(i, j)];
			}
		}

		out
	}
}

impl<T: Copy + Scalar, const N: usize> From<OVector<T, Const<N>>>
	for Vector<T, N>
{
	fn from(value: OVector<T, Const<N>>) -> Self {
		value.data.0[0].into()
	}
}
