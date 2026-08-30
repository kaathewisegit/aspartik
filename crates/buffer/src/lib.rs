mod index;
mod raw;
mod slice_buf;

pub use index::Size;
pub use raw::RawBuffer;
pub use slice_buf::SliceBuffer;

use bytemuck::AnyBitPattern;

use std::{
	ops::{Deref, DerefMut},
	ptr::drop_in_place,
};

pub struct Buffer<T, Idx: Size = u32, const ALIGN: usize = 0> {
	raw: RawBuffer<T, ALIGN>,
	len: Idx,
}

// SAFETY: it's only an allocation
unsafe impl<T, Idx: Size, const ALIGN: usize> Send for Buffer<T, Idx, ALIGN> where
	T: Send
{
}
// SAFETY: same as above
unsafe impl<T, Idx: Size, const ALIGN: usize> Sync for Buffer<T, Idx, ALIGN> where
	T: Sync
{
}

impl<T, Idx: Size, const ALIGN: usize> Buffer<T, Idx, ALIGN> {
	pub const ALIGNMENT: usize = RawBuffer::<T>::ALIGNMENT;

	pub fn new(len: Idx) -> Self
	where
		T: AnyBitPattern,
	{
		assert_ne!(len, Idx::ZERO, "Length must be bigger than 0");
		// SAFETY: `len` checked above
		let raw = unsafe { RawBuffer::uninit(len) };

		Self { raw, len }
	}

	pub fn repeat(value: T, len: Idx) -> Self
	where
		T: Copy,
	{
		assert_ne!(len, Idx::ZERO, "Length must be bigger than 0");
		// SAFETY: `len` checked above, we'll initialize all of the
		// elements below
		let raw = unsafe { RawBuffer::uninit(len) };

		for i in 0..len.usize() {
			// SAFETY: `i < len`
			unsafe { *raw.get_mut(i) = value };
		}

		Self { raw, len }
	}

	pub fn len(&self) -> Idx {
		self.len
	}
}

impl<T, Idx: Size, const ALIGN: usize> Drop for Buffer<T, Idx, ALIGN> {
	fn drop(&mut self) {
		// SAFETY: `raw` has a length of `len`
		unsafe { drop_in_place(self.raw.as_raw_slice_mut(self.len)) }

		// SAFETY: `raw` has a length of `len`
		unsafe { self.raw.deallocate(self.len) };
	}
}

impl<T, Idx: Size, const ALIGN: usize> Deref for Buffer<T, Idx, ALIGN> {
	type Target = [T];

	fn deref(&self) -> &[T] {
		// SAFETY: `raw` has a length of `len`
		unsafe { &*self.raw.as_raw_slice(self.len) }
	}
}

impl<T, Idx: Size, const ALIGN: usize> DerefMut for Buffer<T, Idx, ALIGN> {
	fn deref_mut(&mut self) -> &mut [T] {
		// SAFETY: `raw` has a length of `len`
		unsafe { &mut *self.raw.as_raw_slice_mut(self.len) }
	}
}
