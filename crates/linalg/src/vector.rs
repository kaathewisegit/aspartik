use num_traits::Num;

use std::ops::{Index, IndexMut};

pub trait Vector<T: Copy + Num, const N: usize>:
	Sized + Index<usize, Output = T> + IndexMut<usize>
{
	/// Create a new vector by repeating (splatting) `value`
	fn from_element(value: T) -> Self;

	/// Creates a vector of zeros
	fn zeros() -> Self {
		Self::from_element(T::zero())
	}

	/// Map the vector into a new one
	fn map<F, U, V>(&self, f: F) -> V
	where
		F: FnMut(T) -> U,
		U: Num + Copy,
		V: Vector<U, N>;

	/// Edit every element in place
	fn for_each<F>(&mut self, f: F)
	where
		F: FnMut(&mut T);

	/// Per-element product of two vectors
	fn hadamard<V: Vector<T, N>>(&self, rhs: impl Vector<T, N>) -> V {
		let mut out = V::zeros();
		for i in 0..N {
			out[i] = self[i] * rhs[i];
		}
		out
	}

	/// Sum of all elements in a vector
	fn sum(&self) -> T {
		let mut out = self[0];
		for i in 1..N {
			out = out + self[i];
		}
		out
	}

	/// Product of all elements in a vector
	fn product(&self) -> T {
		let mut out = self[0];
		for i in 1..N {
			out = out * self[i];
		}
		out
	}

	/// Dot product of two vectors which can have different types
	fn dot_product(&self, other: &impl Vector<T, N>) -> T {
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

	fn map<F, U, V>(&self, mut f: F) -> V
	where
		F: FnMut(T) -> U,
		U: Num + Copy,
		V: Vector<U, N>,
	{
		let mut out = V::zeros();
		for i in 0..N {
			out[i] = f(self[i]);
		}
		out
	}

	fn for_each<F>(&mut self, f: F)
	where
		F: FnMut(&mut T),
	{
		self.iter_mut().for_each(f)
	}
}
