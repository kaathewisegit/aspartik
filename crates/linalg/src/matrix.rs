use bytemuck::Zeroable;
use num_traits::{Num, One};

use std::{
	marker::PhantomData,
	ops::{Deref, Index, IndexMut},
	ptr,
};

use crate::Dim;

#[repr(C)]
pub struct MatrixRef<'a, T> {
	ptr: *const T,
	dim: Dim,
	marker: PhantomData<&'a T>,
}

impl<T> Clone for MatrixRef<'_, T> {
	fn clone(&self) -> Self {
		*self
	}
}

impl<T> Copy for MatrixRef<'_, T> {}

impl<'a, T> MatrixRef<'a, T> {
	/// Create a new matrix reference from a pointer
	///
	/// # Safety
	///
	/// `ptr` must point to an allocation which lives for at least `'a`, has
	/// no mutable references to it for the duration of the existence of a
	/// `MatrixRef`, and has at least `dim.num_slots()` valid elements after
	/// `ptr`.
	pub unsafe fn from_raw_parts(ptr: *const T, dim: Dim) -> Self {
		Self {
			ptr,
			dim,
			marker: PhantomData,
		}
	}

	pub fn from_array<const N: usize, const M: usize>(
		arr: &'a [[T; M]; N],
	) -> Self {
		Self {
			ptr: arr.as_ptr() as *const T,
			dim: Dim {
				rows: N as u32,
				cols: M as u32,
				row_stride: M as u32,
			},
			marker: PhantomData,
		}
	}

	pub fn from_slice(slice: &'a [T], rows: usize, cols: usize) -> Self {
		assert!(slice.len() <= rows * cols);
		Self {
			ptr: slice.as_ptr(),
			dim: Dim {
				rows: rows as u32,
				cols: cols as u32,
				row_stride: rows as u32,
			},
			marker: PhantomData,
		}
	}

	pub fn num_rows(self) -> usize {
		self.dim.rows as usize
	}

	pub fn num_cols(self) -> usize {
		self.dim.cols as usize
	}

	pub fn is_square(self) -> bool {
		self.dim.is_square()
	}

	pub fn to_boxed_slice(self) -> Box<[T]>
	where
		T: Copy,
	{
		let mut out = Vec::with_capacity(self.num_elements());
		for i in 0..self.num_rows() {
			for j in 0..self.num_cols() {
				out.push(self[(i, j)]);
			}
		}
		out.into_boxed_slice()
	}

	pub fn num_elements(&self) -> usize {
		self.num_rows() * self.num_cols()
	}

	/// Get a reference to the element at the index `(row, col)`
	///
	/// # Safety
	///
	/// - `row < self.num_rows()`
	/// - `col < self.num_cols()`
	pub unsafe fn at_unchecked(self, row: usize, col: usize) -> &'a T {
		debug_assert!(self.dim.is_index_valid(row, col));
		// SAFETY: unsafe function with invariants
		unsafe { &*self.ptr.add(self.dim.offset(row, col)) }
	}
}

impl<'a, T, const N: usize, const M: usize> From<&'a [[T; M]; N]>
	for MatrixRef<'a, T>
{
	fn from(arr: &'a [[T; M]; N]) -> Self {
		Self::from_array(arr)
	}
}

impl<T> Index<(usize, usize)> for MatrixRef<'_, T> {
	type Output = T;

	fn index(&self, (row, col): (usize, usize)) -> &T {
		assert!(self.dim.is_index_valid(row, col));
		// SAFETY: invariant checked above
		unsafe { self.at_unchecked(row, col) }
	}
}

#[repr(C)]
pub struct MatrixMut<'a, T> {
	ptr: *mut T,
	dim: Dim,
	marker: PhantomData<&'a mut T>,
}

impl<'a, T> MatrixMut<'a, T> {
	pub fn from_array<const N: usize, const M: usize>(
		arr: &'a mut [[T; M]; N],
	) -> Self {
		Self {
			ptr: arr.as_mut_ptr() as *mut T,
			dim: Dim {
				rows: N as u32,
				cols: M as u32,
				row_stride: M as u32,
			},
			marker: PhantomData,
		}
	}

	pub fn from_slice(
		slice: &'a mut [T],
		rows: usize,
		cols: usize,
	) -> Self {
		assert!(slice.len() <= rows * cols);
		Self {
			ptr: slice.as_mut_ptr(),
			dim: Dim {
				rows: rows as u32,
				cols: cols as u32,
				row_stride: cols as u32,
			},
			marker: PhantomData,
		}
	}

	pub fn reborrow<'b>(&'b mut self) -> MatrixMut<'b, T> {
		MatrixMut {
			ptr: self.ptr,
			dim: self.dim,
			marker: PhantomData,
		}
	}

	/// Zeroes out the matrix
	pub fn zero(self)
	where
		T: Zeroable,
	{
		// SAFETY: The matrix points to an allocation at least
		// `num_slots` large.
		unsafe { ptr::write_bytes(self.ptr, 0, self.dim.num_slots()) }
	}

	pub fn identity(mut self)
	where
		T: Zeroable + One,
	{
		assert!(self.reborrow().is_square());
		self.reborrow().zero();

		for i in 0..self.num_cols() {
			self[(i, i)] = T::one();
		}
	}

	/// Sets the diagonal elements to `values`
	///
	/// Does not zero out other elements of the matrix.
	pub fn set_diagonal(mut self, values: &[T])
	where
		T: Copy,
	{
		assert!(self.reborrow().is_square());
		for i in 0..self.num_rows() {
			self[(i, i)] = values[i];
		}
	}

	pub fn swap_rows(self, a: usize, b: usize) {
		assert!(a < self.num_rows());
		assert!(b < self.num_rows());
		if a == b {
			return;
		}

		let row_stride = self.dim.row_stride as usize;
		// SAFETY: `a` checked above
		let ptr_a = unsafe { self.ptr.add(row_stride * a) };
		// SAFETY: `b` checked above
		let ptr_b = unsafe { self.ptr.add(row_stride * b) };

		// SAFETY: `a` != `b`, each slice has a length of `rows`, so
		// they don't overlap.
		unsafe {
			ptr::swap_nonoverlapping(ptr_a, ptr_b, self.num_rows())
		}
	}

	/// Get a mutable reference to the element at the index `(row, col)`
	///
	/// # Safety
	///
	/// - `row < self.num_rows()`
	/// - `col < self.num_cols()`
	pub unsafe fn at_mut_unchecked(
		self,
		row: usize,
		col: usize,
	) -> &'a mut T {
		debug_assert!(self.dim.is_index_valid(row, col));
		// SAFETY: unsafe function with invariants
		unsafe { &mut *self.ptr.add(self.dim.offset(row, col)) }
	}

	pub fn copy_from(mut self, src: MatrixRef<'_, T>)
	where
		T: Copy,
	{
		for i in 0..self.num_rows() {
			for j in 0..self.num_cols() {
				self[(i, j)] = src[(i, j)];
			}
		}
	}
}

impl<'a, T, const N: usize, const M: usize> From<&'a mut [[T; M]; N]>
	for MatrixMut<'a, T>
{
	fn from(arr: &'a mut [[T; M]; N]) -> Self {
		Self::from_array(arr)
	}
}

impl<'a, T> Deref for MatrixMut<'a, T> {
	type Target = MatrixRef<'a, T>;

	fn deref(&self) -> &Self::Target {
		// SAFETY: `MatrixRef` has the same layout as `MatrixMut`,
		// except for `ptr` and `marker`.  `*const T` has the same
		// layout as `*mut T`.  And the marker has mutable and immutable
		// references, but both of those are covariant.
		unsafe { &*(self as *const Self as *const Self::Target) }
	}
}

impl<T> Index<(usize, usize)> for MatrixMut<'_, T> {
	type Output = T;

	fn index(&self, (row, col): (usize, usize)) -> &T {
		assert!(self.dim.is_index_valid(row, col));
		// SAFETY: invariant checked above
		unsafe { self.at_unchecked(row, col) }
	}
}

impl<T> IndexMut<(usize, usize)> for MatrixMut<'_, T> {
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
		assert!(self.dim.is_index_valid(row, col));
		// SAFETY: invariant checked above
		unsafe { self.reborrow().at_mut_unchecked(row, col) }
	}
}

pub fn mul<'a, T, L, R, D>(lhs: L, rhs: R, dst: D)
where
	T: Copy + Num + Zeroable + 'a,
	L: Into<MatrixRef<'a, T>>,
	R: Into<MatrixRef<'a, T>>,
	D: Into<MatrixMut<'a, T>>,
{
	mul_inner(lhs.into(), rhs.into(), dst.into())
}

pub fn mul_inner<T>(
	lhs: MatrixRef<'_, T>,
	rhs: MatrixRef<'_, T>,
	mut dst: MatrixMut<'_, T>,
) where
	T: Copy + Num + Zeroable,
{
	assert_eq!(lhs.num_cols(), rhs.num_rows());
	assert_eq!(dst.num_rows(), lhs.num_rows());
	assert_eq!(dst.num_cols(), rhs.num_cols());

	let rows = dst.num_rows();
	let cols = dst.num_cols();
	let inner = lhs.num_cols();

	for i in 0..rows {
		for j in 0..cols {
			let mut sum = T::zeroed();
			for k in 0..inner {
				sum = sum + lhs[(i, k)] * rhs[(k, j)];
			}
			dst[(i, j)] = sum;
		}
	}
}
