use std::{
	alloc::{Layout, alloc, dealloc, handle_alloc_error},
	ptr::{NonNull, slice_from_raw_parts, slice_from_raw_parts_mut},
};

use super::Size;

/// A contiguous array allocation for `T`
///
/// `RawBuffer` doesn't contain its own capacity, so it should be tracked by
/// its upstream user.
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

	fn layout<Idx: Size>(capacity: Idx) -> Layout {
		Layout::from_size_align(
			capacity.usize() * size_of::<T>(),
			Self::ALIGNMENT,
		)
		.expect("`capacity` too big")
	}

	/// Allocate a new uninitialized slice of size `capacity`.
	///
	/// # Safety
	///
	/// - `capacity` must not be zero
	pub unsafe fn uninit<Idx: Size>(capacity: Idx) -> Self {
		#[expect(unused)]
		{
			Self::_CHECK_ALIGN_POW2;
			Self::_CHECK_ALIGN_SIZE;
			Self::_NON_ZST;
		}

		debug_assert_ne!(capacity, Idx::ZERO);
		let layout = Self::layout(capacity);

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
	/// `idx * size_of::<T>()` must not overflow `isize`.
	///
	/// Additionally, the resulting pointer is only valid if `idx` is
	/// within the buffer's capacity.
	pub unsafe fn get<Idx: Size>(&self, idx: Idx) -> *const T {
		// SAFETY: function's unsafe invariant
		unsafe { self.ptr.add(idx.usize()) }.as_ptr()
	}

	/// # Safety
	///
	/// `idx * size_of::<T>()` must not overflow `isize`.
	pub unsafe fn get_mut<Idx: Size>(&self, idx: Idx) -> *mut T {
		// SAFETY: same as `get`
		(unsafe { self.get(idx) }) as *mut T
	}

	pub fn ptr(&self) -> NonNull<T> {
		self.ptr
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

	/// Get a prefix slice of length `len` pointing into the buffer
	///
	/// # Safety
	///
	/// `len` must be less than or equal to the buffer capacity.
	/// Violating this triggers undefined behavior even if the elements
	/// past the end of the buffer are never accessed.
	pub unsafe fn as_slice<Idx: Size>(&mut self, len: Idx) -> &[T] {
		// SAFETY: invariant
		unsafe { &*self.as_raw_slice(len) }
	}

	/// Mutable prefix-slice
	///
	/// # Safety
	///
	/// `len` must be less than or equal to the buffer capacity.
	pub unsafe fn as_slice_mut<Idx: Size>(&mut self, len: Idx) -> &mut [T] {
		// SAFETY: invariant
		unsafe { &mut *self.as_raw_slice_mut(len) }
	}

	/// Deallocate the buffer
	///
	/// # Safety
	///
	/// The `capacity` must be correct.
	///
	/// Additionally, while Rust allows leaking all types, not dropping the
	/// contents of the buffer might lead to correctness issues.
	pub unsafe fn deallocate<Idx: Size>(&mut self, capacity: Idx) {
		let layout = Self::layout(capacity);
		// SAFETY: capacity (and thus layout) should be valid by the
		// function invariant.
		unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
	}
}
