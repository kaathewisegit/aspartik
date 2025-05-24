use num_traits::{Num, NumAssign};

use std::{
	fmt::{self, Display},
	ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign},
};

use crate::vector::Vector;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct RowMatrix<T: Copy, const N: usize, const M: usize> {
	m: [Vector<T, N>; M],
}

// Constructors
impl<T: Copy, const N: usize, const M: usize> From<[Vector<T, N>; M]>
	for RowMatrix<T, N, M>
{
	fn from(value: [Vector<T, N>; M]) -> Self {
		RowMatrix { m: value }
	}
}

impl<T: Copy, const N: usize, const M: usize> From<[[T; N]; M]>
	for RowMatrix<T, N, M>
{
	fn from(value: [[T; N]; M]) -> Self {
		value.map(|row| -> Vector<T, N> { row.into() }).into()
	}
}

impl<T, const N: usize, const M: usize> Display for RowMatrix<T, N, M>
where
	T: Copy + Display,
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("[")?;

		for i in 0..N {
			for j in 0..M {
				self[(i, j)].fmt(f)?;

				if !(i == N - 1 && j == M - 1) {
					f.write_str(",")?;
				}
				if j != M - 1 {
					f.write_str(" ")?;
				}
			}

			if i != N - 1 {
				f.write_str("\n ")?;
			}
		}

		f.write_str("]")?;

		Ok(())
	}
}

// Numerical constructors
impl<T: Copy + Num, const N: usize, const M: usize> RowMatrix<T, N, M> {
	/// A matrix with all of it's elements set to zero.
	pub fn zeros() -> Self {
		Self::from_element(T::zero())
	}

	/// A matrix with all of it's elements set to one.  Not that useful, but
	/// still here to make use of the `One` `Num` subtrait.
	pub fn ones() -> Self {
		Self::from_element(T::one())
	}
}

impl<T: Copy + Num, const N: usize> RowMatrix<T, N, N> {
	/// Identity matrix.
	pub fn identity() -> Self {
		let mut out = Self::zeros();
		for i in 0..N {
			out[(i, i)] = T::one();
		}
		out
	}

	pub fn from_diagonal(diag: Vector<T, N>) -> Self {
		let mut out = Self::zeros();

		for i in 0..N {
			out[(i, i)] = diag[i];
		}

		out
	}
}

impl<T: Copy, const N: usize, const M: usize> RowMatrix<T, N, M> {
	/// Creates a new matrix with all of it's elements set to `value`.
	pub fn from_element(value: T) -> Self {
		[Vector::from([value; N]); M].into()
	}
}

impl<T: Copy + Default, const N: usize, const M: usize> Default
	for RowMatrix<T, N, M>
{
	fn default() -> Self {
		Self::from_element(T::default())
	}
}

// Operators

impl<T: Copy, const N: usize, const M: usize> Index<usize>
	for RowMatrix<T, N, M>
{
	type Output = Vector<T, N>;

	fn index(&self, i: usize) -> &Vector<T, N> {
		&self.m[i]
	}
}

impl<T: Copy, const N: usize, const M: usize> IndexMut<usize>
	for RowMatrix<T, N, M>
{
	fn index_mut(&mut self, i: usize) -> &mut Vector<T, N> {
		&mut self.m[i]
	}
}

impl<T: Copy, const N: usize, const M: usize> Index<(usize, usize)>
	for RowMatrix<T, N, M>
{
	type Output = T;

	fn index(&self, (i, j): (usize, usize)) -> &T {
		&self.m[i][j]
	}
}

impl<T: Copy, const N: usize, const M: usize> IndexMut<(usize, usize)>
	for RowMatrix<T, N, M>
{
	fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
		&mut self.m[i][j]
	}
}

impl<T: Copy + AddAssign, const N: usize, const M: usize> AddAssign
	for RowMatrix<T, N, M>
{
	fn add_assign(&mut self, rhs: Self) {
		for i in 0..M {
			self[i] += rhs[i];
		}
	}
}

impl<T: Copy + AddAssign, const N: usize, const M: usize> Add
	for RowMatrix<T, N, M>
{
	type Output = Self;

	fn add(mut self, rhs: Self) -> Self {
		for i in 0..M {
			self[i] += rhs[i];
		}
		self
	}
}

impl<T: Copy + MulAssign, const N: usize, const M: usize> RowMatrix<T, N, M> {
	pub fn component_mul_assign(&mut self, rhs: Self) {
		for i in 0..M {
			self[i] *= rhs[i];
		}
	}

	pub fn component_mul(mut self, rhs: Self) -> Self {
		for i in 0..M {
			self[i] *= rhs[i];
		}
		self
	}
}

impl<T: Copy + AddAssign, const N: usize, const M: usize> RowMatrix<T, N, M> {
	pub fn trace(&self) -> T {
		let mut out = self[(0, 0)];
		for i in 1..N {
			out += self[(i, i)];
		}
		out
	}
}

impl<T, const N: usize, const M: usize> Mul<Vector<T, N>> for RowMatrix<T, N, M>
where
	T: Copy + NumAssign + Default,
{
	type Output = Vector<T, M>;

	fn mul(self, rhs: Vector<T, N>) -> Vector<T, M> {
		// TODO: uninitialized
		let mut out = Vector::default();

		for i in 0..M {
			out[i] = (self[i] * rhs).sum();
		}

		out
	}
}

impl<T, const N: usize, const M: usize> Mul<T> for RowMatrix<T, N, M>
where
	T: Copy + MulAssign,
{
	type Output = Self;

	fn mul(self, rhs: T) -> Self::Output {
		let mut out = self;

		for i in 0..N {
			for j in 0..M {
				out[(i, j)] *= rhs;
			}
		}

		out
	}
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<RowMatrix<T, M, P>>
	for RowMatrix<T, N, M>
where
	T: Copy + AddAssign + Mul<Output = T> + Default,
{
	type Output = RowMatrix<T, N, P>;

	// This is a suboptimal algorithm.  There's a more cache-friendly one,
	// but it requires calculating a bunch of things, including a square
	// root.  I should implement both, compare the assembly output, and
	// benchmark.
	//
	// https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
	fn mul(self, rhs: RowMatrix<T, M, P>) -> Self::Output {
		let mut out = RowMatrix::default();

		for i in 0..N {
			for j in 0..P {
				for k in 0..M {
					out[(i, j)] +=
						self[(i, k)] * rhs[(k, j)];
				}
			}
		}

		out
	}
}

// Type-agnostic implementations.
impl<T: Copy, const N: usize, const M: usize> RowMatrix<T, N, M> {
	pub const NUM_ROWS: usize = N;
	pub const NUM_COLUMNS: usize = M;
	pub const NUM_ITEMS: usize = N * M;
	pub const IS_SQUARE: bool = N == M;

	pub fn as_mut_ptr(&mut self) -> *mut T {
		self[0].as_mut_ptr()
	}

	pub fn as_ptr(&self) -> *const T {
		self[0].as_ptr()
	}

	pub fn apply<F>(&mut self, f: F)
	where
		F: Fn(&mut T),
	{
		self.m.each_mut().map(|v| v.apply(&f));
	}

	pub fn map<F, U>(&self, f: F) -> RowMatrix<U, N, M>
	where
		U: Copy,
		F: Fn(T) -> U,
	{
		self.m.map(|v| v.map(&f)).into()
	}

	pub fn map_diagonal<F>(&self, f: F) -> RowMatrix<T, N, M>
	where
		F: Fn(T) -> T,
	{
		let mut out = *self;

		for i in 0..std::cmp::min(N, M) {
			out[(i, i)] = f(out[(i, i)]);
		}

		out
	}

	pub fn truncate<const NX: usize, const MX: usize>(
		&self,
	) -> RowMatrix<T, NX, MX> {
		assert!(NX < N, "New length must be smaller");
		assert!(MX < M, "New length must be smaller");

		let subslice: &[Vector<T, N>; MX] =
			self.m.first_chunk().unwrap();

		subslice.map(|v| v.truncate()).into()
	}

	pub fn transpose(&self) -> RowMatrix<T, M, N>
	where
		T: Default,
	{
		let mut out = RowMatrix::default();

		for i in 0..N {
			for j in 0..M {
				out[j][i] = self[i][j];
			}
		}

		out
	}
}

// Numeric methods.
impl<T: Copy, const N: usize> RowMatrix<T, N, N> {
	pub fn is_symmetric(&self) -> bool
	where
		T: PartialEq,
	{
		for i in 0..N {
			for j in (i + 1)..N {
				if self[i][j] != self[j][i] {
					return false;
				}
			}
		}

		true
	}
}
