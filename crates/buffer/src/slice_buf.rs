use std::ops::{Index, IndexMut, Range};

use bytemuck::AnyBitPattern;

use crate::{Buffer, Size};

pub struct SliceBuffer<T, Idx: Size = usize, const ALIGN: usize = 0> {
	buffer: Buffer<T, Idx, ALIGN>,
	size: Idx,
}

impl<T, Idx: Size, const ALIGN: usize> SliceBuffer<T, Idx, ALIGN> {
	/// Creates a buffer capable of holding `len` slices of `size`
	pub fn new(size: Idx, len: Idx) -> Self
	where
		T: AnyBitPattern,
	{
		Self {
			buffer: Buffer::new(size * len),
			size,
		}
	}

	pub fn len(&self) -> Idx {
		self.buffer.len() / self.size
	}

	fn range(&self, index: Idx) -> Range<usize> {
		let size = self.size.usize();
		let index = index.usize();
		let offset = size * index;
		offset..offset + size
	}

	/// # Safety
	///
	/// `index` must be less than `self.len()`
	pub unsafe fn get_unchecked(&self, index: Idx) -> &[T] {
		// SAFETY: per the function invariant `range(index)` must be
		// valid
		unsafe { self.buffer.get_unchecked(self.range(index)) }
	}

	/// # Safety
	///
	/// `index` must be less than `self.len()`
	pub unsafe fn get_mut_unchecked(&mut self, index: Idx) -> &mut [T] {
		let range = self.range(index);
		// SAFETY: per the function invariant `range(index)` must be
		// valid
		unsafe { self.buffer.get_unchecked_mut(range) }
	}
}

impl<T, Idx: Size, const ALIGN: usize> Index<Idx>
	for SliceBuffer<T, Idx, ALIGN>
{
	type Output = [T];

	fn index(&self, index: Idx) -> &[T] {
		&self.buffer[self.range(index)]
	}
}

impl<T, Idx: Size, const ALIGN: usize> IndexMut<Idx>
	for SliceBuffer<T, Idx, ALIGN>
{
	fn index_mut(&mut self, index: Idx) -> &mut [T] {
		let range = self.range(index);
		&mut self.buffer[range]
	}
}
