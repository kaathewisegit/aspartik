use std::{
	alloc::{Layout, alloc, dealloc, handle_alloc_error},
	ptr::{NonNull, slice_from_raw_parts, slice_from_raw_parts_mut},
};

use super::Size;

pub struct RawBuffer<T, const ALIGN: usize = 0> {
	ptr: NonNull<T>,
}

// SAFETY: it's only an allocation
unsafe impl<T, const ALIGN: usize> Send for RawBuffer<T, ALIGN> where T: Send {}
// SAFETY: same as above
unsafe impl<T, const ALIGN: usize> Sync for RawBuffer<T, ALIGN> where T: Sync {}

impl<T, const ALIGN: usize> RawBuffer<T, ALIGN> {
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

	/// Allocate a new uninitialized slice of size `len`.
	///
	/// # Safety
	///
	/// - `len` must not be zero
	pub unsafe fn uninit<Idx: Size>(len: Idx) -> Self {
		#[expect(unused)]
		{
			Self::_CHECK_ALIGN_POW2;
			Self::_CHECK_ALIGN_SIZE;
			Self::_NON_ZST;
		}

		debug_assert_ne!(len, Idx::ZERO);
		let layout = Layout::from_size_align(
			len.usize() * size_of::<T>(),
			Self::ALIGNMENT,
		)
		.expect("`len` too big");

		// SAFETY: we've checked above that capacity/size isn't 0 and
		// there's a const assertion that size_of::<T> isn't 0.
		let ptr = unsafe { alloc(layout) as *mut T };
		let Some(ptr) = NonNull::new(ptr) else {
			handle_alloc_error(layout);
		};

		Self { ptr }
	}

	/// # Safety
	///
	/// `len * size_of::<T>()` must not overflow `isize`.
	pub unsafe fn get<Idx: Size>(&self, len: Idx) -> *const T {
		// SAFETY: function's unsafe invariant
		unsafe { self.ptr.add(len.usize()) }.as_ptr()
	}

	/// # Safety
	///
	/// `len * size_of::<T>()` must not overflow `isize`.
	pub unsafe fn get_mut<Idx: Size>(&self, len: Idx) -> *mut T {
		// SAFETY: same as `get`
		(unsafe { self.get(len) }) as *mut T
	}

	pub fn as_raw_slice_mut<Idx: Size>(&mut self, len: Idx) -> *mut [T] {
		// SAFETY (from `slice::from_raw_parts_mut`)
		// - `ptr` is non-null, valid, and points to an allocation
		//   `capacity * size_of::<T>()` large.
		// - The data is aligned
		// - The reference is unique
		// - Total size is less than `isize::MAX`, checked by
		//   `Layout::from_size_align`
		slice_from_raw_parts_mut(self.ptr.as_ptr(), len.usize())
	}

	pub fn as_raw_slice<Idx: Size>(&self, len: Idx) -> *const [T] {
		slice_from_raw_parts(self.ptr.as_ptr(), len.usize())
	}

	/// Deallocate the slice
	///
	/// # Safety
	///
	/// `len` must have the same value as one passed to `uninit`.
	pub unsafe fn drop<Idx: Size>(&mut self, len: Idx) {
		let layout = Layout::from_size_align(
			len.usize() * size_of::<T>(),
			Self::ALIGNMENT,
		)
		.unwrap();

		// SAFETY: the layout is the same
		unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) }
	}

	/// Prefix-slice
	///
	/// # Safety
	///
	/// `len` must be less or equal to one passed in `uninit`.
	pub unsafe fn as_slice<Idx: Size>(&mut self, len: Idx) -> &[T] {
		// SAFETY: invariant
		unsafe { &*self.as_raw_slice(len) }
	}

	/// Mutable prefix-slice
	///
	/// # Safety
	///
	/// `len` must be less or equal to one passed in `uninit`.
	pub unsafe fn as_slice_mut<Idx: Size>(&mut self, len: Idx) -> &[T] {
		// SAFETY: invariant
		unsafe { &*self.as_raw_slice_mut(len) }
	}
}
