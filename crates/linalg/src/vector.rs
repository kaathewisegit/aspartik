use num_traits::Num;

use std::ops::{Index, IndexMut};

pub trait Vector<T: Copy, const N: usize>:
	Index<usize, Output = T> + IndexMut<usize>
{
	fn from_element(value: T) -> Self;

	fn map<F, U: Copy>(&self, f: F) -> impl Vector<U, N>
	where
		F: FnMut(T) -> U;

	fn for_each<F>(&mut self, f: F)
	where
		F: FnMut(&mut T);

	// truncate?
}

pub trait VectorNum<T, const N: usize>: Vector<T, N>
where
	T: Copy + Num,
{
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

	fn dot_product(&self, other: impl VectorNum<T, N>) -> T {
		let mut out = self[0] * other[0];

		for i in 1..N {
			out = out + self[i] * other[i];
		}

		out
	}
}

impl<T: Copy, const N: usize> Vector<T, N> for [T; N] {
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
