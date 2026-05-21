use bytemuck::Zeroable;
use num_traits::{Num, One};

use std::{
	marker::PhantomData,
	ops::{Deref, Index, IndexMut},
	ptr,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatrixRef<'a, T> {
	ptr: *const T,
	rows: u32,
	cols: u32,
	stride: u32,
	marker: PhantomData<&'a T>,
}

impl<'a, T: Copy> MatrixRef<'a, T> {
	pub fn from_array<const N: usize, const M: usize>(
		arr: &'a [[T; M]; N],
	) -> Self {
		Self {
			ptr: arr.as_ptr() as *const T,
			rows: N as u32,
			cols: M as u32,
			stride: M as u32,
			marker: PhantomData,
		}
	}

	pub fn from_slice(slice: &'a [T], rows: usize, cols: usize) -> Self {
		assert!(slice.len() <= rows * cols);
		Self {
			ptr: slice.as_ptr(),
			rows: rows as u32,
			cols: cols as u32,
			stride: cols as u32,
			marker: PhantomData,
		}
	}

	pub fn num_rows(self) -> usize {
		self.rows as usize
	}

	pub fn num_cols(self) -> usize {
		self.cols as usize
	}

	pub fn is_square(self) -> bool {
		self.cols == self.rows
	}

	pub fn to_boxed_slice(self) -> Box<[T]> {
		let mut out = Vec::with_capacity(self.num_elements());
		for i in 0..self.rows {
			for j in 0..self.cols {
				out.push(self[(i, j)]);
			}
		}
		out.into_boxed_slice()
	}

	pub fn num_elements(&self) -> usize {
		(self.rows * self.cols) as usize
	}

	/// Total number of item slots, including padding
	fn num_slots(&self) -> usize {
		(self.rows * self.stride) as usize
	}
}

impl<'a, T: Copy, const N: usize, const M: usize> From<&'a [[T; M]; N]>
	for MatrixRef<'a, T>
{
	fn from(arr: &'a [[T; M]; N]) -> Self {
		Self::from_array(arr)
	}
}

impl<T> Index<(usize, usize)> for MatrixRef<'_, T> {
	type Output = T;

	fn index(&self, (row, col): (usize, usize)) -> &T {
		assert!(row < self.rows as usize && col < self.cols as usize);
		let idx = row * self.stride as usize + col;
		// SAFETY: TODO
		unsafe { &*self.ptr.add(idx) }
	}
}

impl<T> Index<(u32, u32)> for MatrixRef<'_, T> {
	type Output = T;

	fn index(&self, (row, col): (u32, u32)) -> &T {
		assert!(row < self.rows && col < self.cols);
		let idx = row * self.stride + col;
		// SAFETY: TODO
		unsafe { &*self.ptr.add(idx as usize) }
	}
}

#[repr(C)]
pub struct MatrixMut<'a, T> {
	ptr: *mut T,
	rows: u32,
	cols: u32,
	stride: u32,
	marker: PhantomData<&'a mut T>,
}

impl<'a, T: Copy> MatrixMut<'a, T> {
	pub fn from_array<const N: usize, const M: usize>(
		arr: &'a mut [[T; M]; N],
	) -> Self {
		Self {
			ptr: arr.as_mut_ptr() as *mut T,
			rows: N as u32,
			cols: M as u32,
			stride: M as u32,
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
			rows: rows as u32,
			cols: cols as u32,
			stride: cols as u32,
			marker: PhantomData,
		}
	}

	pub fn reborrow<'b>(&'b mut self) -> MatrixMut<'b, T> {
		MatrixMut {
			ptr: self.ptr,
			rows: self.rows,
			cols: self.cols,
			stride: self.stride,
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
		unsafe { ptr::write_bytes(self.ptr, 0, self.num_slots()) }
	}

	pub fn identity(mut self)
	where
		T: Zeroable + One,
	{
		assert!(self.reborrow().is_square());
		self.reborrow().zero();

		for i in 0..self.cols {
			self[(i, i)] = T::one();
		}
	}

	pub fn swap_rows(self, a: usize, b: usize) {
		assert!(a < self.rows as usize);
		assert!(b < self.rows as usize);
		if a == b {
			return;
		}

		// SAFETY: `a` checked above
		let ptr_a = unsafe { self.ptr.add(self.stride as usize * a) };
		// SAFETY: `b` checked above
		let ptr_b = unsafe { self.ptr.add(self.stride as usize * b) };

		// SAFETY: `a` != `b`, each slice has a length of `rows`, so
		// they don't overlap.
		unsafe {
			ptr::swap_nonoverlapping(
				ptr_a,
				ptr_b,
				self.rows as usize,
			)
		}
	}
}

impl<'a, T: Copy, const N: usize, const M: usize> From<&'a mut [[T; M]; N]>
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
		assert!(row < self.rows as usize && col < self.cols as usize);
		let idx = row * self.stride as usize + col;
		// SAFETY: TODO
		unsafe { &*self.ptr.add(idx) }
	}
}

impl<T> IndexMut<(usize, usize)> for MatrixMut<'_, T> {
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
		assert!(row < self.rows as usize && col < self.cols as usize);
		// SAFETY: TODO
		unsafe { &mut *self.ptr.add(row * self.stride as usize + col) }
	}
}

impl<T> Index<(u32, u32)> for MatrixMut<'_, T> {
	type Output = T;

	fn index(&self, (row, col): (u32, u32)) -> &T {
		assert!(row < self.rows && col < self.cols);
		let idx = row * self.stride + col;
		// SAFETY: TODO
		unsafe { &*self.ptr.add(idx as usize) }
	}
}

impl<T> IndexMut<(u32, u32)> for MatrixMut<'_, T> {
	fn index_mut(&mut self, (row, col): (u32, u32)) -> &mut T {
		assert!(row < self.rows && col < self.cols);
		let idx = row * self.stride + col;
		// SAFETY: TODO
		unsafe { &mut *self.ptr.add(idx as usize) }
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
