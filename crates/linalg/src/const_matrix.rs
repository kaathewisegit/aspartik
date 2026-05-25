use std::marker::PhantomData;

use num_traits::Num;

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

pub trait ConstMatrix<T, const N: usize, const M: usize>:
	Sized + Clone
where
	T: Copy + Num,
{
	fn at(&self, row: usize, column: usize) -> &T;

	fn at_mut(&mut self, row: usize, column: usize) -> &mut T;

	fn from_element(value: T) -> Self;

	fn for_each<F>(&mut self, f: F)
	where
		F: FnMut(&mut T);

	fn convert<O>(&self) -> O
	where
		O: ConstMatrix<T, N, M>,
	{
		let mut out = O::zeros();
		for i in 0..N {
			for j in 0..M {
				*out.at_mut(i, j) = *self.at(i, j);
			}
		}
		out
	}

	fn zeros() -> Self {
		Self::from_element(T::zero())
	}

	fn transpose<O>(&self) -> O
	where
		O: ConstMatrix<T, M, N>,
	{
		let mut out = O::zeros();

		for i in 0..N {
			for j in 0..M {
				*out.at_mut(j, i) = *self.at(i, j);
			}
		}

		out
	}

	fn mul<const P: usize, O>(&self, rhs: &impl ConstMatrix<T, M, P>) -> O
	where
		O: ConstMatrix<T, N, P>,
	{
		let mut out = O::zeros();

		for i in 0..N {
			for j in 0..P {
				for k in 0..M {
					*out.at_mut(i, j) = *out.at(i, j)
						+ *self.at(i, k)
							* *rhs.at(k, j);
				}
			}
		}

		out
	}

	fn mul_scalar<O>(&self, rhs: T) -> O
	where
		O: ConstMatrix<T, N, M>,
	{
		let mut out = O::zeros();

		for i in 0..N {
			for j in 0..M {
				*out.at_mut(i, j) = *self.at(i, j) * rhs;
			}
		}

		out
	}

	fn add<O>(&self, rhs: &impl ConstMatrix<T, N, M>) -> O
	where
		O: ConstMatrix<T, N, M>,
	{
		let mut out = O::zeros();

		for i in 0..N {
			for j in 0..M {
				*out.at_mut(i, j) =
					*self.at(i, j) + *rhs.at(i, j);
			}
		}

		out
	}

	fn sub<O>(&self, rhs: &impl ConstMatrix<T, N, M>) -> O
	where
		O: ConstMatrix<T, N, M>,
	{
		let mut out = O::zeros();

		for i in 0..N {
			for j in 0..M {
				*out.at_mut(i, j) =
					*self.at(i, j) - *rhs.at(i, j);
			}
		}

		out
	}

	fn swap_rows(&mut self, a: usize, b: usize) {
		for col in 0..M {
			(*self.at_mut(a, col), *self.at_mut(b, col)) =
				(*self.at(b, col), *self.at(a, col));
		}
	}
}

pub trait ConstSquareMatrix<T, const N: usize>: ConstMatrix<T, N, N>
where
	T: Copy + Num,
{
	fn trace(&self) -> T {
		let mut out = *self.at(0, 0);
		for i in 1..N {
			out = out + *self.at(i, i);
		}
		out
	}
}

impl<T, const N: usize, M> ConstSquareMatrix<T, N> for M
where
	T: Copy + Num,
	M: ConstMatrix<T, N, N>,
{
}

impl<T, const N: usize, const M: usize> ConstMatrix<T, N, M> for [[T; M]; N]
where
	T: Copy + Num,
{
	fn at(&self, row: usize, column: usize) -> &T {
		&self[row][column]
	}

	fn at_mut(&mut self, row: usize, column: usize) -> &mut T {
		&mut self[row][column]
	}

	fn from_element(value: T) -> Self {
		[[value; M]; N]
	}

	fn for_each<F>(&mut self, mut f: F)
	where
		F: FnMut(&mut T),
	{
		for i in 0..N {
			for j in 0..M {
				f(self.at_mut(i, j));
			}
		}
	}
}
