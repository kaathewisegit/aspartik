//! # SkBuf
//!
//! SkBuf is an epoch-versioned [`Vec`]-like structure with epoch versioning.
//! It's designed for branchless value access and memory locality between the
//! data versions.
//!
//! The API mostly mirrors that of [`Vec`].  New vectors can be created using
//! the [`skvec!`] macro, which has the same syntax as [`vec!`].  Value access
//! can be done via indexing.  Due to implementation details `SkBuf` doesn't
//! implement [`IndexMut`][std::ops::IndexMut], so value updates have to be done
//! with [`set`][SkBuf::set].
//!
//! The core feature, versioning, can be used via two methods.
//!
//! - [`accept`][SkBuf::accept] confirms all of the edits done since the last
//!   epoch and drops the overwritten items.
//!
//! - [`reject`][SkBuf::reject] rolls back all of the elements to the values
//!   they had at the start of the last epoch.
//!
//! Where an epoch is the time of creation of the vector or the last call to
//! `accept` or `reject`.  For the precise terminology (i.e. the difference
//! between elements and items) see the [`SkBuf`] type documentation.
//!
//!
//! ## Example
//!
//! ```
//! use sk::{skbuf, SkBuf};
//!
//! let mut v = skbuf![1, 2, 3];
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
use serde::{Deserialize, Serialize};

use std::ops::Index;

/// Epoch-versioned `Vec`-like storage.
///
/// `SkBuf` is made up of *elements*.  Each element is addressable by its index
/// and is made out of two *items*.  The first item is the original value of the
/// element in a single epoch.  The second one is the new, edited value, created
/// with [`set`][SkBuf::set].  On [`accept`][SkBuf::accept] the second item will
/// become the primary one and the old one will be erased.  And on
/// [`reject`][SkBuf::reject] the second item will be erased, with the element
/// falling back to the original one.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SkBuf<T> {
	/// The underlying storage.  It's twice as long as the number of items
	/// `SkBuf` can hold at a time.  Each element consist of two items in
	/// `inner`, only one of which is active, determined by the `mask` at
	/// the index.
	items: Vec<T>,
	/// Metadata associated with each element
	///
	/// - The first bit is a pointer to the first or the second element.
	/// - The second bit is the edited state: 0 if not edited, 1 if edited.
	metadata: Vec<u8>,
}

// Memoization-related methods
impl<T> SkBuf<T> {
	/// Returns the offset, which is always 0 or 1
	///
	/// # Safety
	///
	/// `i` must be less than the length of `self`.
	unsafe fn offset(&self, i: usize) -> usize {
		// SAFETY: `i < self.len()` invariant
		let m = unsafe { self.metadata.get_unchecked(i) } & 0b1;
		usize::from(m)
	}

	/// Returns the currently active item at index `i`.
	///
	/// # Safety
	///
	/// `i` must be less than the length of `self`.
	pub unsafe fn get_unchecked(&self, i: usize) -> &T {
		// SAFETY: `i < self.len()` invariant
		let idx = i * 2 + unsafe { self.offset(i) };
		// SAFETY: `i < self.len()`, so `i * 2 + 1` is less than
		// `self.len() * 2`, the length of `inner`
		unsafe { self.items.get_unchecked(idx) }
	}

	/// Mutable version of [`active_inner`][SkBuf::active_inner].
	///
	/// # Safety
	///
	/// `i` must be less than the length of `self`.
	pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
		// SAFETY: `i < self.len()` invariant
		let idx = i * 2 + unsafe { self.offset(i) };
		// SAFETY: see `active_inner`
		unsafe { self.items.get_unchecked_mut(idx) }
	}

	fn is_edited(&self, i: usize) -> bool {
		(self.metadata[i] & 0b10) != 0
	}

	/// Accept all of the changes made since the creation of the vector or
	/// the last call to `accept` or [`reject`][SkBuf::reject].
	pub fn accept(&mut self) {
		// zero-out the edited status
		for m in &mut self.metadata {
			*m &= 0b01;
		}
	}

	/// Reject all of the changes made this epoch.
	pub fn reject(&mut self) {
		for m in &mut self.metadata {
			// 00 -> 00
			// 01 -> 01
			// 10 -> 01
			// 11 -> 00
			*m = (*m ^ (*m >> 1)) & 1;
		}
	}

	/// Version of `set` without bounds checking
	///
	/// # Safety
	///
	/// `index` must be less than the length of `self`.
	pub unsafe fn set_unchecked(&mut self, index: usize, value: T) {
		// - If edited is 0, we set it to 1 and flip offset
		// - If edited is 1, we keep it and keep the offset
		// 00 -> 11
		// 01 -> 10
		// 10 -> 10
		// 11 -> 11
		// SAFETY: `index < self.len()` invariant
		let m = unsafe { self.metadata.get_unchecked_mut(index) };
		*m = ((*m & 0b01) ^ !(*m >> 1)) & 0b11;

		// SAFETY: `index < self.len()` invariant
		let item = unsafe { self.get_unchecked_mut(index) };
		*item = value;
	}

	/// Sets the item at `index` to `value`.  All of the subsequent index
	/// operations (via [`SkBuf::index`] or the `[]` operator) will return
	/// the updated item which equals value.
	pub fn set(&mut self, index: usize, value: T) {
		assert!(index < self.len());
		// SAFETY: invariant checked above
		unsafe { self.set_unchecked(index, value) }
	}

	/// Returns `true` if at least a single element has been changed
	///
	/// This only accounts for `set` calls, not values.  So, if an element
	/// is overwritten with the same value, `is_changed` will still return
	/// `true`.
	pub fn is_changed(&self) -> bool {
		self.metadata.iter().any(|&e| (e & 0b10) != 0)
	}

	/// Checks if the element at `index` has been edited during this epoch
	pub fn is_changed_at(&self, index: usize) -> bool {
		(self.metadata[index] & 0b10) != 0
	}
}

// Trait implementations

impl<T> Index<usize> for SkBuf<T> {
	type Output = T;

	fn index(&self, index: usize) -> &T {
		assert!(index < self.len());
		// SAFETY: the invariant is checked above
		unsafe { self.get_unchecked(index) }
	}
}

// Iterator implementations

/// Immutable iterator over a [`SkBuf`].
///
/// See [`SkBuf::iter`].
pub struct Iter<'a, T> {
	vec: &'a SkBuf<T>,
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

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.len(), Some(self.len()))
	}

	fn count(self) -> usize
	where
		Self: Sized,
	{
		self.len()
	}

	fn last(self) -> Option<Self::Item>
	where
		Self: Sized,
	{
		if self.index == self.vec.len() {
			None
		} else {
			self.vec.last()
		}
	}
}

impl<T> ExactSizeIterator for Iter<'_, T> {
	fn len(&self) -> usize {
		self.vec.len() - self.index
	}
}

impl<T> SkBuf<T> {
	/// Returns an iterator over the vector, which yields currently active
	/// item values.
	pub fn iter(&self) -> Iter<'_, T> {
		Iter {
			vec: self,
			index: 0,
		}
	}
}

impl<'a, T> IntoIterator for &'a SkBuf<T> {
	type Item = &'a T;
	type IntoIter = Iter<'a, T>;

	fn into_iter(self) -> Iter<'a, T> {
		self.iter()
	}
}

// Methods from `Vec`.
impl<T> SkBuf<T> {
	/// Creates a new `SkVec` with `size` default elements
	pub fn new(length: usize) -> Self
	where
		T: Default + Clone,
	{
		let metadata = vec![0; length];
		let items = vec![T::default(); length * 2];

		Self { items, metadata }
	}

	/// Number of items in the `SkBuf`.
	///
	/// See [`SkBuf` documentation][SkBuf] for the distinction between items
	/// and values.
	pub fn len(&self) -> usize {
		self.metadata.len()
	}

	/// Returns `true` if the vector has no items.
	pub fn is_empty(&self) -> bool {
		self.items.is_empty()
	}

	/// Returns the last active element, or `None` if the vector is empty.
	pub fn last(&self) -> Option<&T> {
		if self.is_empty() {
			None
		} else {
			Some(&self[self.len() - 1])
		}
	}

	/// Convert active items to a contiguous vector
	pub fn to_vec(&self) -> Vec<T>
	where
		T: Clone,
	{
		let mut out = Vec::with_capacity(self.len());

		for i in 0..self.len() {
			out.push(self[i].clone());
		}

		out
	}
}

// Custom
impl<T> SkBuf<T> {
	/// Constructs a vector made out of `value` repeated `length` times.
	pub fn repeat(value: T, length: usize) -> Self
	where
		T: Clone,
	{
		let metadata = vec![0; length];
		let mut items = Vec::with_capacity(length * 2);

		for _ in 0..length {
			items.push(value.clone());
			items.push(value.clone());
		}

		Self { items, metadata }
	}
}

// From implementations

impl<T: Clone> From<&[T]> for SkBuf<T> {
	fn from(values: &[T]) -> Self {
		let metadata = vec![0; values.len()];
		let mut items = Vec::with_capacity(values.len() * 2);

		for value in values {
			items.push(value.clone());
			items.push(value.clone());
		}

		Self { items, metadata }
	}
}

impl<T: Clone> From<Vec<T>> for SkBuf<T> {
	fn from(values: Vec<T>) -> Self {
		let metadata = vec![0; values.len()];
		let mut items = Vec::with_capacity(values.len() * 2);

		for value in values {
			items.push(value.clone());
			items.push(value);
		}

		Self { items, metadata }
	}
}

impl<T: Clone, const N: usize> From<[T; N]> for SkBuf<T> {
	fn from(values: [T; N]) -> Self {
		let metadata = vec![0; values.len()];
		let mut items = Vec::with_capacity(values.len() * 2);

		for value in values {
			items.push(value.clone());
			items.push(value);
		}

		Self { items, metadata }
	}
}

/// Works identically to [`vec!`].
#[macro_export]
macro_rules! skbuf {
	($elem:expr; $n:expr) => {
		$crate::SkBuf::repeat($elem, $n)
	};
	($($x:expr),+ $(,)?) => {
		$crate::SkBuf::from([$($x),+])
	}
}
