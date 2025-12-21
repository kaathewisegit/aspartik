use num_traits::{Float, Num, NumAssign};

use std::{
	cmp::Ordering,
	fmt::{self, Debug, Display},
	ops::{
		Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign,
	},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct Vector<T, const N: usize> {
	v: [T; N],
}

// `From` conversions
impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
	fn from(value: [T; N]) -> Self {
		Self { v: value }
	}
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
	fn from(value: Vector<T, N>) -> Self {
		value.v
	}
}

impl<T: Copy, const N: usize> Vector<T, N> {
	pub fn from_element(value: T) -> Self {
		[value; N].into()
	}
}

impl<T: Debug, const N: usize> Debug for Vector<T, N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("[")?;

		for i in 0..N {
			self[i].fmt(f)?;
			if i != N - 1 {
				f.write_str(", ")?;
			}
		}

		f.write_str("]")?;

		Ok(())
	}
}

impl<T: Display, const N: usize> Display for Vector<T, N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("[")?;

		for i in 0..N {
			self[i].fmt(f)?;
			if i != N - 1 {
				f.write_str(", ")?;
			}
		}

		f.write_str("]")?;

		Ok(())
	}
}

// This can be derived, but then every single trait impl would have to carry a
// `Default` constraint around, which is verbose.
impl<T: Copy + Default, const N: usize> Default for Vector<T, N> {
	fn default() -> Self {
		[T::default(); N].into()
	}
}

// Mathematical constructors
impl<T: Copy + Num, const N: usize> Vector<T, N> {
	pub fn zeros() -> Self {
		Self::from_element(T::zero())
	}

	pub fn ones() -> Self {
		Self::from_element(T::one())
	}

	/// A standard basis vector: all elements are zero, except the one at
	/// index `i`, which is set to one.
	pub fn sbv(i: usize) -> Self {
		let mut out = Self::zeros();
		out[i] = T::one();
		out
	}
}

// Operators and overloading in general.
impl<T, const N: usize> Index<usize> for Vector<T, N> {
	type Output = T;

	fn index(&self, index: usize) -> &Self::Output {
		&self.v[index]
	}
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.v[index]
	}
}

impl<T: Copy + AddAssign, const N: usize> AddAssign for Vector<T, N> {
	fn add_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self[i] += rhs[i];
		}
	}
}

impl<T: Copy + AddAssign, const N: usize> Add for Vector<T, N> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		let mut out = self;
		out += rhs;
		out
	}
}

impl<T: Copy + MulAssign, const N: usize> MulAssign for Vector<T, N> {
	fn mul_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self[i] *= rhs[i];
		}
	}
}

impl<T: Copy + MulAssign, const N: usize> Mul for Vector<T, N> {
	type Output = Self;

	fn mul(mut self, rhs: Self) -> Self::Output {
		self *= rhs;
		self
	}
}

impl<T: Copy + MulAssign, const N: usize> MulAssign<T> for Vector<T, N> {
	fn mul_assign(&mut self, rhs: T) {
		for i in 0..N {
			self[i] *= rhs;
		}
	}
}

impl<T: Copy + MulAssign, const N: usize> Mul<T> for Vector<T, N> {
	type Output = Self;

	fn mul(mut self, rhs: T) -> Self::Output {
		self *= rhs;
		self
	}
}

impl<T: Copy + DivAssign, const N: usize> DivAssign for Vector<T, N> {
	fn div_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self[i] /= rhs[i];
		}
	}
}

impl<T: Copy + DivAssign, const N: usize> Div for Vector<T, N> {
	type Output = Self;

	fn div(mut self, rhs: Self) -> Self::Output {
		self /= rhs;
		self
	}
}

impl<T: Copy + DivAssign, const N: usize> DivAssign<T> for Vector<T, N> {
	fn div_assign(&mut self, rhs: T) {
		for i in 0..N {
			self[i] /= rhs;
		}
	}
}

impl<T: Copy + DivAssign, const N: usize> Div<T> for Vector<T, N> {
	type Output = Self;

	fn div(mut self, rhs: T) -> Self::Output {
		self /= rhs;
		self
	}
}

impl<T: Copy + PartialEq, const N: usize> PartialEq<T> for Vector<T, N> {
	fn eq(&self, other: &T) -> bool {
		for i in 0..N {
			if self[i] != *other {
				return false;
			}
		}
		true
	}
}

impl<T: Copy + PartialEq, const N: usize> PartialEq<[T; N]> for Vector<T, N> {
	fn eq(&self, other: &[T; N]) -> bool {
		for i in 0..N {
			if self[i] != other[i] {
				return false;
			}
		}
		true
	}
}

impl<T: Copy + PartialEq, const N: usize> PartialEq<Vector<T, N>> for [T; N] {
	fn eq(&self, other: &Vector<T, N>) -> bool {
		other == self
	}
}

impl<T: Copy + PartialOrd, const N: usize> PartialOrd<T> for Vector<T, N> {
	fn partial_cmp(&self, other: &T) -> Option<Ordering> {
		let result = self[0].partial_cmp(other)?;
		for i in 1..N {
			let new_result = self[i].partial_cmp(other)?;
			if new_result != result {
				return None;
			}
		}
		Some(result)
	}

	fn lt(&self, other: &T) -> bool {
		for i in 0..N {
			if self[i] >= *other {
				return false;
			}
		}
		true
	}
}

// Type-agnostic implementations.
impl<T, const N: usize> Vector<T, N> {
	pub const LENGTH: usize = N;

	pub fn as_ptr(&self) -> *const T {
		self.v.as_ptr()
	}

	pub fn as_mut_ptr(&mut self) -> *mut T {
		self.v.as_mut_ptr()
	}

	pub fn as_array(&self) -> &[T; N] {
		&self.v
	}

	pub fn as_mut_array(&mut self) -> &mut [T; N] {
		&mut self.v
	}

	pub fn to_array(self) -> [T; N] {
		self.v
	}

	pub fn apply<F>(&mut self, f: F)
	where
		F: Fn(&mut T),
	{
		for element in self.v.each_mut() {
			f(element);
		}
	}

	pub fn map<F, U>(&self, f: F) -> Vector<U, N>
	where
		U: Copy,
		F: Fn(T) -> U,
		T: Copy,
	{
		self.v.map(f).into()
	}

	pub fn truncate<const M: usize>(&self) -> Vector<T, M>
	where
		T: Copy,
	{
		assert!(M < N, "New length must be smaller");

		let subslice: &[T; M] = self.v.first_chunk().unwrap();

		Vector::from(*subslice)
	}
}

// Numeric methods.
impl<T: Copy + NumAssign, const N: usize> Vector<T, N> {
	pub fn sum(&self) -> T {
		let mut out = self[0];
		for i in 1..N {
			out += self[i];
		}
		out
	}

	pub fn product(&self) -> T {
		let mut out = self[0];
		for i in 1..N {
			out *= self[i];
		}
		out
	}

	pub fn dot_product(&self, other: &Vector<T, N>) -> T {
		let mut out = self[0] * other[0];

		for i in 1..N {
			out += self[i] * other[i];
		}

		out
	}
}

impl<T: Float + NumAssign, const N: usize> Vector<T, N> {
	pub fn magnitude(&self) -> T {
		self.dot_product(self).sqrt()
	}

	pub fn consine_similarity(&self, other: &Vector<T, N>) -> T {
		self.dot_product(other) / (self.magnitude() * other.magnitude())
	}
}

impl<T: Copy + Float + NumAssign, const N: usize> Vector<T, N> {
	pub fn norm(&self) -> T {
		let mut out = self[0] * self[0];

		for i in 1..N {
			out += self[i] * self[i];
		}

		out.sqrt()
	}

	pub fn normalize(&self) -> Self {
		*self / self.norm()
	}

	pub fn minimum(&self) -> T {
		let mut out = self[0];

		for i in 1..N {
			out = out.min(self[i]);
		}

		out
	}

	pub fn maximum(&self) -> T {
		let mut out = self[0];

		for i in 1..N {
			out = out.max(self[i]);
		}

		out
	}
}
