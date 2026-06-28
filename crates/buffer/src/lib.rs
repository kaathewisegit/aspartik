mod slice_buf;

pub use slice_buf::SliceBuffer;

use bytemuck::AnyBitPattern;

use std::{
	alloc::{Layout, alloc, dealloc, handle_alloc_error},
	ops::{Deref, DerefMut},
	ptr::{
		NonNull, drop_in_place, slice_from_raw_parts,
		slice_from_raw_parts_mut,
	},
};

pub struct Buffer<T, const ALIGN: usize = 0> {
	ptr: NonNull<T>,
	len: usize,
}

// SAFETY: it's only an allocation
unsafe impl<T, const ALIGN: usize> Send for Buffer<T, ALIGN> where T: Send {}
// SAFETY: same as above
unsafe impl<T, const ALIGN: usize> Sync for Buffer<T, ALIGN> where T: Sync {}

macro_rules! check_const {
	() => {
		#[expect(unused)]
		{
			Self::_CHECK_ALIGN_POW2;
			Self::_CHECK_ALIGN_SIZE;
			Self::_NON_ZST;
		}
	};
}

impl<T, const ALIGN: usize> Buffer<T, ALIGN> {
	pub const ALIGNMENT: usize =
		{ if ALIGN == 0 { align_of::<T>() } else { ALIGN } };

	/// ```compile_fail
	/// buffer::Buffer::<u8, 3>::new(10);
	/// ```
	const _CHECK_ALIGN_POW2: () = assert!(
		ALIGN == 0 || ALIGN.is_power_of_two(),
		"ALIGN must be a power of two"
	);

	/// ```compile_fail
	/// buffer::Buffer::<u64, 2>::new(10);
	/// ```
	const _CHECK_ALIGN_SIZE: () = assert!(
		ALIGN == 0 || ALIGN >= align_of::<T>(),
		"ALIGN be equal to or bigger than `T`'s required alignment"
	);

	/// ```compile_fail
	/// buffer::Buffer::<(), 3>::new(10);
	/// ```
	const _NON_ZST: () =
		assert!(size_of::<T>() != 0, "T must not be a ZST");

	pub fn new(len: usize) -> Self
	where
		T: AnyBitPattern,
	{
		check_const!();

		assert_ne!(len, 0, "Length must be bigger than 0");
		let layout = Layout::from_size_align(
			len * size_of::<T>(),
			Self::ALIGNMENT,
		)
		.expect("`len` too big");

		// SAFETY: we've checked above that capacity/size isn't 0 and
		// there's a const assertion that size_of::<T> isn't 0.
		let ptr = unsafe { alloc(layout) as *mut T };
		let Some(ptr) = NonNull::new(ptr) else {
			handle_alloc_error(layout);
		};

		Self { ptr, len }
	}

	pub fn repeat(value: T, len: usize) -> Self
	where
		T: Copy + AnyBitPattern,
	{
		let mut out = Self::new(len);
		for element in &mut *out {
			*element = value;
		}
		out
	}

	fn as_raw_mut_slice(&mut self) -> *mut [T] {
		// SAFETY (from `slice::from_raw_parts_mut`)
		// - `ptr` is non-null, valid, and points to an allocation
		//   `capacity * size_of::<T>()` large.
		// - The data is aligned
		// - The reference is unique
		// - Total size is less than `isize::MAX`, checked by
		//   `Layout::from_size_align`
		slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len)
	}

	fn as_raw_slice(&self) -> *const [T] {
		// SAFETY: see `as_raw_mut_slice`
		slice_from_raw_parts(self.ptr.as_ptr(), self.len)
	}
}

impl<T, const ALIGN: usize> Drop for Buffer<T, ALIGN> {
	fn drop(&mut self) {
		let layout = Layout::from_size_align(
			self.len * size_of::<T>(),
			Self::ALIGNMENT,
		)
		.unwrap();

		// SAFETY: `as_raw_mut_slice` is valid
		unsafe { drop_in_place(self.as_raw_mut_slice()) }

		// SAFETY: the layout is the same
		unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) }
	}
}

impl<T, const ALIGN: usize> Deref for Buffer<T, ALIGN> {
	type Target = [T];

	fn deref(&self) -> &[T] {
		// SAFETY: `as_raw_slice` is valid, see its safety
		unsafe { &*self.as_raw_slice() }
	}
}

impl<T, const ALIGN: usize> DerefMut for Buffer<T, ALIGN> {
	fn deref_mut(&mut self) -> &mut [T] {
		// SAFETY: `as_raw_mut_slice` is valid, see its safety
		unsafe { &mut *self.as_raw_mut_slice() }
	}
}

impl<T, const ALIGN: usize> Clone for Buffer<T, ALIGN>
where
	T: Copy + AnyBitPattern,
{
	fn clone(&self) -> Self {
		let mut out = Self::new(self.len());
		out.copy_from_slice(self);
		out
	}
}

impl<T, const ALIGN: usize> From<&[T]> for Buffer<T, ALIGN>
where
	T: Copy + AnyBitPattern,
{
	fn from(value: &[T]) -> Self {
		let mut out = Self::new(value.len());
		out.copy_from_slice(value);
		out
	}
}
