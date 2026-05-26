use std::ops::{Index, IndexMut, Range};

use bytemuck::Zeroable;

use crate::Buffer;

pub struct SliceBuffer<T, const ALIGN: usize = 0> {
	buffer: Buffer<T, ALIGN>,
	size: usize,
}

impl<T, const ALIGN: usize> SliceBuffer<T, ALIGN> {
	/// Creates a buffer capable of holding `len` slices of `size`
	pub fn new(size: usize, len: usize) -> Self
	where
		T: Zeroable,
	{
		Self {
			buffer: Buffer::new(size * len),
			size,
		}
	}

	#[expect(clippy::len_without_is_empty)]
	pub fn len(&self) -> usize {
		self.buffer.len() / self.size
	}

	fn range(&self, index: usize) -> Range<usize> {
		let offset = self.size * index;
		offset..offset + self.size
	}

	/// # Safety
	///
	/// `index` must be less than `self.len()`
	pub unsafe fn get_unchecked(&self, index: usize) -> &[T] {
		// SAFETY: per the function invariant `range(index)` must be
		// valid
		unsafe { self.buffer.get_unchecked(self.range(index)) }
	}

	/// # Safety
	///
	/// `index` must be less than `self.len()`
	pub unsafe fn get_mut_unchecked(&mut self, index: usize) -> &mut [T] {
		let range = self.range(index);
		// SAFETY: per the function invariant `range(index)` must be
		// valid
		unsafe { self.buffer.get_unchecked_mut(range) }
	}
}

impl<T, const ALIGN: usize> Index<usize> for SliceBuffer<T, ALIGN> {
	type Output = [T];

	fn index(&self, index: usize) -> &[T] {
		&self.buffer[self.range(index)]
	}
}

impl<T, const ALIGN: usize> IndexMut<usize> for SliceBuffer<T, ALIGN> {
	fn index_mut(&mut self, index: usize) -> &mut [T] {
		let range = self.range(index);
		&mut self.buffer[range]
	}
}
