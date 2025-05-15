//! # SkVec
//!
//! SkVec is an epoch-versioned [`Vec`]-like structure with epoch versioning.
//! It's designed for branchless value access and memory locality between the
//! data versions.
//!
//! The API mostly mirrors that of [`Vec`].  New vectors can be created using
//! the [`skvec!`] macro, which has the same syntax as [`vec!`].  Value access
//! can be done via indexing.  Due to implementation details `SkVec` doesn't
//! implement [`IndexMut`][std::ops::IndexMut], so value updates have to be done
//! with [`set`][SkVec::set].
//!
//! The core feature, versioning, can be used via two methods.
//!
//! - [`accept`][SkVec::accept] confirms all of the edits done since the last
//!   epoch and drops the overwritten items.
//!
//! - [`reject`][SkVec::reject] rolls back all of the elements to the values
//!   they had at the start of the last epoch.
//!
//! Where an epoch is the time of creation of the vector or the last call to
//! `accept` or `reject`.  For the precise terminology (i.e. the difference
//! between elements and items) see the [`SkVec`] type documentation.
//!
//!
//! ## Example
//!
//! ```
//! use skvec::{skvec, SkVec};
//!
//! let mut v = skvec![1, 2, 3];
//! assert_eq!(v, [1, 2, 3]);
//!
//! v.set(0, 10);
//! v.set(2, 30);
//! assert_eq!(v, [10, 2, 30]);
//!
//! v.accept();
//! assert_eq!(v, [10, 2, 30]);
//!
//! v.set(1, 20);
//! assert_eq!(v, [10, 20, 30]);
//!
//! v.reject();
//! assert_eq!(v, [10, 2, 30]);
//! ```

mod debug;
mod eq;
#[cfg(feature = "serde")]
mod serde;

use std::{
	mem::{needs_drop, MaybeUninit},
	ops::Index,
};

/// Epoch-versioned `Vec`-like storage.
///
/// `SkVec` is made up of *elements*.  Each element is addressable by its index
/// and is made out of two *items*.  The first item is the original value of the
/// element in a single epoch.  The second one is the new, edited value, created
/// with [`set`][SkVec::set].  On [`accept`][SkVec::accept] the second item will
/// become the primary one and the old one will be erased.  And on
/// [`reject`][SkVec::reject] the second item will be erased, with the element
/// falling back to the original one.
pub struct SkVec<T> {
	/// The underlying storage.  It's twice as long as the number of items
	/// `SkVec` can hold at a time.  Each element consist of two items in
	/// `inner`, only one of which is active, determined by the `mask` at
	/// the index.
	inner: Vec<MaybeUninit<T>>,
	/// True if an element had been edited.  It uses the `bool` type, which
	/// is guaranteed to be one byte:
	///
	/// - <https://doc.rust-lang.org/std/mem/fn.size_of.html#:~:text=bool>
	/// - <https://github.com/rust-lang/rust/pull/46156>
	edited: Vec<bool>,
	/// Mask points to the currently active item in `inner`.
	///
	/// ## Safety
	///
	/// - Masks must have the values of either 0 or 1.
	///
	/// - Masks must point at initialized memory.  When an element is
	///   created for the first time, the 0th one is presumed to be
	///   initialized.  This can change after an `accept` call.
	mask: Vec<u8>,
}

// Memoization-related methods
impl<T> SkVec<T> {
	/// Returns the currently active item at index `i`.
	fn active_inner(&self, i: usize) -> &MaybeUninit<T> {
		&self.inner[i * 2 + self.mask[i] as usize]
	}

	/// Mutable version of [`active_inner`][SkVec::active_inner].
	fn active_inner_mut(&mut self, i: usize) -> &mut MaybeUninit<T> {
		&mut self.inner[i * 2 + self.mask[i] as usize]
	}

	/// The other slot at index `i`, which is not being pointed to by the
	/// mask.
	fn inactive_inner_mut(&mut self, i: usize) -> &mut MaybeUninit<T> {
		&mut self.inner[i * 2 + (self.mask[i] ^ 1) as usize]
	}

	/// Drops all of the active items.
	///
	/// # Safety
	///
	/// Calling this function breaks the `inner` invariant because it
	/// deinitializes all active elements.
	unsafe fn deinit(&mut self) {
		for i in 0..self.len() {
			// SAFETY: masks must always point to initialized
			// values.
			unsafe {
				self.active_inner_mut(i).assume_init_drop();
			}
			if self.edited[i] {
				// SAFETY: since the element has been edited,
				// the inactive element is the original one, so
				// it must be initialized.
				unsafe {
					self.inactive_inner_mut(i)
						.assume_init_drop()
				}
			}
		}
	}

	/// Zero-out the edited status array.
	fn clear_edited(&mut self) {
		self.edited.iter_mut().for_each(|v| *v = false);
	}

	#[cfg(feature = "serde")]
	fn first_item(&self, index: usize) -> Option<&T> {
		// first item is either active, or the element has been edited
		// and the second item is active
		if self.mask[index] == 0
			|| (self.mask[index] == 1 && self.edited[index])
		{
			let item = &self.inner[index * 2];
			// SAFETY: we know from the condition that this field
			// is either active or inactive, but edited, meaning it
			// was initialized when the epoch started.
			let item = unsafe { item.assume_init_ref() };
			Some(item)
		} else {
			None
		}
	}

	#[cfg(feature = "serde")]
	fn second_item(&self, index: usize) -> Option<&T> {
		if self.mask[index] == 1
			|| (self.mask[index] == 0 && self.edited[index])
		{
			let item = &self.inner[index * 2 + 1];
			// SAFETY: same as `first_item`
			let item = unsafe { item.assume_init_ref() };
			Some(item)
		} else {
			None
		}
	}

	/// Accept all of the changes made since the creation of the vector or
	/// the last call to `accept` or [`reject`][SkVec::reject].
	///
	/// If `T` is [`Drop`], all of the overwritten elements will be dropped,
	/// which will take awhile for long arrays.  If `T` is not [`Drop`],
	/// this method much faster.
	pub fn accept(&mut self) {
		// Don't compile the loop for non-drop types.
		if needs_drop::<T>() {
			for i in 0..self.len() {
				if self.edited[i] {
					// SAFETY: Only initialized elements can
					// be edited.  Since the element at
					// index `i` has been updated, it
					// must've been initialized.
					unsafe {
						self.inactive_inner_mut(i)
							.assume_init_drop();
					}
				}
			}
		}

		self.clear_edited();
	}

	/// Reject all of the changes made this epoch.  All edited items will be
	/// dropped and the items will roll back to their old values.
	///
	/// This method is much slower than `accept` for non-[`Drop`] types, as
	/// it has to iterate over the vector to search for edited elements.
	pub fn reject(&mut self) {
		// Don't compile the loop for non-drop types.
		if needs_drop::<T>() {
			for i in 0..self.len() {
				if self.edited[i] {
					// SAFETY: only initialized elements can
					// be edited.  Since the element at
					// index `i` has been updated, it
					// must've been initialized.
					unsafe {
						self.active_inner_mut(i)
							.assume_init_drop();
					}
				}
			}
		}

		for i in 0..self.len() {
			if self.edited[i] {
				// Point back to the old item.
				self.mask[i] ^= 1;
			}
		}

		self.clear_edited();
	}

	/// Sets the item at `index` to `value`.  All of the subsequent index
	/// operations (via [`SkVec::index`] or the `[]` operator) will return
	/// the updated item which equals value.
	pub fn set(&mut self, index: usize, value: T) {
		if self.edited[index] {
			// We are overwriting an older edited item, so drop it.
			//
			// SAFETY: if the item has been edited, it must have a
			// valid value.
			unsafe {
				self.active_inner_mut(index).assume_init_drop();
			}
		} else {
			// The element was unedited, so the item is being
			// written for the first time during this epoch.
			self.mask[index] ^= 1;
			self.edited[index] = true;
		}

		*self.active_inner_mut(index) = MaybeUninit::new(value);
	}

	/// Roll back the item at `index`.
	///
	/// - If the item was edited, this will drop the edited item if needed
	///   and roll back to the old one.  It will not be affected by
	///   subsequent calls to [`accept`][`SkVec::accept`] or
	///   [`reject`][`SkVec::reject`].
	///
	/// - If the item hasn't been edited, this is a no-op.
	///
	/// Essentially, this is an item-local version of `reject`.
	pub fn unset(&mut self, index: usize) {
		if self.edited[index] {
			if needs_drop::<T>() {
				// SAFETY: because `edited[index]` is true, it
				// must've been set before.
				unsafe {
					self.active_inner_mut(index)
						.assume_init_drop();
				}
			}

			self.edited[index] = false;
			self.mask[index] ^= 1;
		}
	}

	/// If the item at `index` has been edited, accept it.
	///
	/// This function acts independently of the `accept` and `reject`
	/// methods.  A subsequent call to either of those won't change the
	/// element or status of the accepted item.
	///
	/// Essentially, this is an item-local version of `accept`.
	pub fn accept_item(&mut self, index: usize) {
		if self.edited[index] {
			if needs_drop::<T>() {
				// SAFETY: the item has been edited, so the
				// inactive slot must've been initialized.
				unsafe {
					self.inactive_inner_mut(index)
						.assume_init_drop();
				}
			}

			self.edited[index] = false;
		}
	}
}

// Trait implementations

impl<T> Drop for SkVec<T> {
	/// Necessary because `MaybeUninit` doesn't drop on deinitialization.
	fn drop(&mut self) {
		if needs_drop::<T>() {
			// SAFETY: all of the items in `inner` are wrapped in
			// `MaybeUninit`, so it's fine to deinitialize them.  No
			// other code will touch the items before `inner` is
			// dropped.
			unsafe { self.deinit() }
		}
	}
}

impl<T> Index<usize> for SkVec<T> {
	type Output = T;

	fn index(&self, index: usize) -> &T {
		// SAFETY:
		//
		// - When an element is added to the vector for the first time,
		//   it's initialized and the mask points at it.
		//
		// - When an element is set, mask is moved to point to the
		//   item.
		//
		// - During `accept` and `reject` the invariant of `mask`
		//   pointing to the initialized items should be preserved.
		//
		// All of that means that this is sound, as long as mutating
		// methods, constructors, `accept`, `reject`, `set`, and in
		// general all of the methods which mutate the vector are sound
		// and uphold the invariants.
		unsafe { self.active_inner(index).assume_init_ref() }
	}
}

impl<T> Clone for SkVec<T>
where
	MaybeUninit<T>: Clone,
{
	fn clone(&self) -> Self {
		Self {
			inner: self.inner.clone(),
			edited: self.edited.clone(),
			mask: self.mask.clone(),
		}
	}
}

impl<T> Default for SkVec<T> {
	fn default() -> Self {
		Self::new()
	}
}

// Iterator implementations

/// Immutable iterator over a [`SkVec`].
///
/// See [`SkVec::iter`].
pub struct Iter<'a, T> {
	vec: &'a SkVec<T>,
	index: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
	type Item = &'a T;

	fn next(&mut self) -> Option<&'a T> {
		if self.index == self.vec.len() {
			None
		} else {
			let out = &self.vec[self.index];
			self.index += 1;
			Some(out)
		}
	}
}

impl<T> SkVec<T> {
	/// Returns an iterator over the vector, which yields currently active
	/// item values.
	pub fn iter(&self) -> Iter<'_, T> {
		Iter {
			vec: self,
			index: 0,
		}
	}
}

impl<'a, T> IntoIterator for &'a SkVec<T> {
	type Item = &'a T;
	type IntoIter = Iter<'a, T>;

	fn into_iter(self) -> Iter<'a, T> {
		self.iter()
	}
}

// Methods from `Vec`.
impl<T> SkVec<T> {
	/// Constructs a new, empty `SkVec`.
	pub fn new() -> Self {
		Self {
			inner: Vec::new(),
			edited: Vec::new(),
			mask: Vec::new(),
		}
	}

	/// Constructs a new, empty `SkVec` which can hold at least `capacity`
	/// elements without additional allocations.
	pub fn with_capacity(capacity: usize) -> Self {
		Self {
			inner: Vec::with_capacity(capacity * 2),
			edited: Vec::with_capacity(capacity),
			mask: Vec::with_capacity(capacity),
		}
	}

	/// Returns the total number of elements the vector can hold without
	/// reallocating.
	///
	/// Note that `SkVec` is made up of several vectors internally, which
	/// are not guaranteed to reserve memory in the same way.  As such,
	/// their capacities might diverge.  This method conservatively returns
	/// the lowest capacity.  Adding more items than that will trigger
	/// allocations, but their exact size might vary in different
	/// situations.
	pub fn capacity(&self) -> usize {
		(self.inner.capacity() / 2)
			.min(self.edited.capacity())
			.min(self.mask.capacity())
	}

	/// Reserve the space for at least `additional` more items.
	///
	/// See [`SkVec::capacity`] for the nuances with handling `SkVec`'s
	/// allocations.
	pub fn reserve(&mut self, additional: usize) {
		self.inner.reserve(additional * 2);
		self.edited.reserve(additional);
		self.mask.reserve(additional);
	}

	/// Shrinks the capacity of the vector as much as possible.
	pub fn shrink_to_fit(&mut self) {
		self.inner.shrink_to_fit();
		self.edited.shrink_to_fit();
		self.mask.shrink_to_fit();
	}

	/// Appends the value as an accepted one.
	pub fn push(&mut self, value: T) {
		self.inner.push(MaybeUninit::new(value));
		self.inner.push(MaybeUninit::uninit());

		self.edited.push(false);
		self.mask.push(0);
	}

	/// Clears the vector, removing all values.
	pub fn clear(&mut self) {
		if needs_drop::<T>() {
			// SAFETY: no code touches the items before they are
			// cleared, so it's fine to drop them.
			unsafe { self.deinit() }
		}

		self.inner.clear();
		self.edited.clear();
		self.mask.clear();
	}

	/// Number of items in the `SkVec`.
	///
	/// See [`SkVec` documentation][SkVec] for the distinction between items
	/// and values.
	pub fn len(&self) -> usize {
		self.mask.len()
	}

	/// Returns `true` if the vector has no items.
	pub fn is_empty(&self) -> bool {
		self.inner.is_empty()
	}

	/// Returns the last active element, or `None` if the vector is empty.
	pub fn last(&self) -> Option<&T> {
		if self.is_empty() {
			None
		} else {
			Some(&self[self.len() - 1])
		}
	}
}

// Custom
impl<T> SkVec<T> {
	/// Constructs a vector made out of `value` repeated `length` times.
	pub fn repeat(value: T, length: usize) -> Self
	where
		T: Clone,
	{
		let mut out = SkVec::with_capacity(length);

		for _ in 0..length {
			out.push(value.clone());
		}

		out
	}
}

// From implementations

impl<T: Clone> From<&[T]> for SkVec<T> {
	fn from(values: &[T]) -> Self {
		let mut out = Self::with_capacity(values.len());

		for value in values {
			out.push(value.clone());
		}

		out
	}
}

impl<T: Clone> From<Vec<T>> for SkVec<T> {
	fn from(value: Vec<T>) -> Self {
		value.as_slice().into()
	}
}

impl<T: Clone, const N: usize> From<[T; N]> for SkVec<T> {
	fn from(values: [T; N]) -> Self {
		let mut out = Self::with_capacity(values.len());

		for value in values {
			out.push(value.clone());
		}

		out
	}
}

/// Works identically to [`vec!`].
#[macro_export]
macro_rules! skvec {
	() => {
		$crate::SkVec::new()
	};
	($elem:expr; $n:expr) => {
		$crate::SkVec::repeat($elem, $n)
	};
	($($x:expr),+ $(,)?) => {
		$crate::SkVec::from([$($x),+])
	}
}
