use num_traits::Num;

use std::ops::{Index, IndexMut};

pub trait Vector<T: Copy + Num, const N: usize>:
	Sized + Index<usize, Output = T> + IndexMut<usize>
{
	fn from_element(value: T) -> Self;

	fn map<F, U: Copy + Num>(&self, f: F) -> impl Vector<U, N>
	where
		F: FnMut(T) -> U;

	fn for_each<F>(&mut self, f: F)
	where
		F: FnMut(&mut T);

	// truncate?

	fn mul(&self, rhs: impl Vector<T, N>) -> Self {
		let mut out = Self::from_element(T::zero());
		for i in 0..N {
			out[i] = self[i] * rhs[i];
		}
		out
	}

	fn sum(&self) -> T {
		let mut out = self[0];
		for i in 1..N {
			out = out + self[i];
		}
		out
	}

	fn product(&self) -> T {
		let mut out = self[0];
		for i in 1..N {
			out = out * self[i];
		}
		out
	}

	fn dot_product(&self, other: impl Vector<T, N>) -> T {
		let mut out = self[0] * other[0];

		for i in 1..N {
			out = out + self[i] * other[i];
		}

		out
	}
}

impl<T: Copy + Num, const N: usize> Vector<T, N> for [T; N] {
	fn from_element(value: T) -> Self {
		[value; N]
	}

	#[expect(refining_impl_trait)]
	fn map<F, U: Copy>(&self, f: F) -> [U; N]
	where
		F: FnMut(T) -> U,
	{
		<[T; N]>::map(*self, f)
	}

	fn for_each<F>(&mut self, f: F)
	where
		F: FnMut(&mut T),
	{
		self.iter_mut().for_each(f)
	}
}
