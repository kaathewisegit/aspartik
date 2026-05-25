use std::marker::PhantomData;

use num_traits::{Num, Zero};

use crate::{Dim, MatrixRef};

#[derive(Clone, Copy)]
pub struct ConstMatrixRef<'a, T, const N: usize, const M: usize> {
	ptr: *const T,
	marker: PhantomData<&'a T>,
}

impl<'a, T, const N: usize, const M: usize> ConstMatrixRef<'a, T, N, M> {
	pub fn from_array(arr: &'a [[T; M]; N]) -> Self {
		Self {
			ptr: arr.as_ptr() as *const T,
			marker: PhantomData,
		}
	}

	pub const fn dim(self) -> Dim {
		Dim {
			rows: N as u32,
			cols: M as u32,
			row_stride: M as u32,
		}
	}

	pub fn as_ref(self) -> MatrixRef<'a, T> {
		// SAFETY: the data will live at least `'a` because of the
		// `self` borrow, and the pointer is valid by construction.
		unsafe { MatrixRef::from_raw_parts(self.ptr, self.dim()) }
	}
}

pub fn from_diagonal<T, const N: usize>(diag: &[T; N]) -> [[T; N]; N]
where
	T: Zero + Copy,
{
	let mut out = [[T::zero(); N]; N];

	for i in 0..N {
		out[i][i] = diag[i];
	}

	out
}

pub fn transpose<T, const N: usize, const M: usize>(
	m: &[[T; M]; N],
) -> [[T; N]; M]
where
	T: Zero + Copy,
{
	let mut trans = [[T::zero(); N]; M];

	for i in 0..N {
		for j in 0..M {
			trans[j][i] = m[i][j];
		}
	}

	trans
}

pub fn mul<T, const N: usize, const M: usize, const K: usize>(
	a: &[[T; M]; N],
	b: &[[T; K]; M],
) -> [[T; K]; N]
where
	T: Num + Copy,
{
	// Initialize the output matrix with T::zero()
	let mut result = [[T::zero(); K]; N];

	for i in 0..N {
		for j in 0..K {
			let mut sum = T::zero();
			for k in 0..M {
				sum = sum + a[i][k] * b[k][j];
			}
			result[i][j] = sum;
		}
	}

	result
}
