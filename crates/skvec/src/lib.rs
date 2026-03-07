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

#![forbid(unsafe_code)]

mod debug;
mod eq;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::ops::Index;

/// Epoch-versioned `Vec`-like storage.
///
/// `SkVec` is made up of *elements*.  Each element is addressable by its index
/// and is made out of two *items*.  The first item is the original value of the
/// element in a single epoch.  The second one is the new, edited value, created
/// with [`set`][SkVec::set].  On [`accept`][SkVec::accept] the second item will
/// become the primary one and the old one will be erased.  And on
/// [`reject`][SkVec::reject] the second item will be erased, with the element
/// falling back to the original one.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SkVec<T> {
	/// The underlying storage.  It's twice as long as the number of items
	/// `SkVec` can hold at a time.  Each element consist of two items in
	/// `inner`, only one of which is active, determined by the `mask` at
	/// the index.
	inner: Vec<T>,
	/// Metadata associated with each element
	///
	/// - The first bit is a pointer to the first or the second element.
	/// - The second bit is the edited state: 0 if not edited, 1 if edited.
	metadata: Vec<u8>,
}

// Memoization-related methods
impl<T> SkVec<T> {
	fn offset(&self, i: usize) -> usize {
		usize::from(self.metadata[i] & 0b1)
	}

	/// Returns the currently active item at index `i`.
	fn active_inner(&self, i: usize) -> &T {
		&self.inner[i * 2 + self.offset(i)]
	}

	/// Mutable version of [`active_inner`][SkVec::active_inner].
	fn active_inner_mut(&mut self, i: usize) -> &mut T {
		let offset = self.offset(i);
		&mut self.inner[i * 2 + offset]
	}

	fn is_edited(&self, i: usize) -> bool {
		(self.metadata[i] & 0b10) != 0
	}

	/// Accept all of the changes made since the creation of the vector or
	/// the last call to `accept` or [`reject`][SkVec::reject].
	///
	/// If `T` is [`Drop`], all of the overwritten elements will be dropped,
	/// which will take awhile for long arrays.  If `T` is not [`Drop`],
	/// this method much faster.
	pub fn accept(&mut self) {
		// zero-out the edited status
		for m in &mut self.metadata {
			*m &= 0b01;
		}
	}

	/// Like `accept`, but only for the `start..end` subslice
	pub fn accept_slice(&mut self, start: usize, end: usize) {
		for m in &mut self.metadata[start..end] {
			*m &= 0b01;
		}
	}

	/// Reject all of the changes made this epoch.  All edited items will be
	/// dropped and the items will roll back to their old values.
	///
	/// This method is much slower than `accept` for non-[`Drop`] types, as
	/// it has to iterate over the vector to search for edited elements.
	pub fn reject(&mut self) {
		for m in &mut self.metadata {
			// 00 -> 00
			// 01 -> 01
			// 10 -> 01
			// 11 -> 00
			*m = (*m ^ (*m >> 1)) & 1;
		}
	}

	/// Like `reject`, but only for the `start..end` subslice
	pub fn reject_slice(&mut self, start: usize, end: usize) {
		for m in &mut self.metadata[start..end] {
			// 00 -> 00
			// 01 -> 01
			// 10 -> 01
			// 11 -> 00
			*m = (*m ^ (*m >> 1)) & 1;
		}
	}

	/// Sets the item at `index` to `value`.  All of the subsequent index
	/// operations (via [`SkVec::index`] or the `[]` operator) will return
	/// the updated item which equals value.
	pub fn set(&mut self, index: usize, value: T) {
		if !self.is_edited(index) {
			// The element was unedited, so the item is being
			// written for the first time during this epoch.

			self.metadata[index] ^= 0b01; // flip the pointer
			self.metadata[index] |= 0b10; // set edited to true
		}

		*self.active_inner_mut(index) = value;
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
	pub fn reject_element(&mut self, index: usize) {
		if self.is_edited(index) {
			self.metadata[index] &= 0b01; // set edited to false
			self.metadata[index] ^= 0b01; // flip the pointer
		}
	}

	/// If the item at `index` has been edited, accept it.
	///
	/// This function acts independently of the `accept` and `reject`
	/// methods.  A subsequent call to either of those won't change the
	/// element or status of the accepted item.
	///
	/// Essentially, this is an item-local version of `accept`.
	pub fn accept_element(&mut self, index: usize) {
		self.metadata[index] &= 0b01;
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
		self.active_inner(index)
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
			metadata: Vec::new(),
		}
	}

	/// Constructs a new, empty `SkVec` which can hold at least `capacity`
	/// elements without additional allocations.
	pub fn with_capacity(capacity: usize) -> Self {
		Self {
			inner: Vec::with_capacity(capacity * 2),
			metadata: Vec::with_capacity(capacity),
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
		(self.inner.capacity() / 2).min(self.metadata.capacity())
	}

	/// Reserve the space for at least `additional` more items.
	///
	/// See [`SkVec::capacity`] for the nuances with handling `SkVec`'s
	/// allocations.
	pub fn reserve(&mut self, additional: usize) {
		self.inner.reserve(additional * 2);
		self.metadata.reserve(additional);
	}

	/// Shrinks the capacity of the vector as much as possible.
	pub fn shrink_to_fit(&mut self) {
		self.inner.shrink_to_fit();
		self.metadata.shrink_to_fit();
	}

	/// Appends the value as an accepted one.
	pub fn push(&mut self, value: T)
	where
		T: Clone,
	{
		self.inner.push(value.clone());
		self.inner.push(value);

		self.metadata.push(0b00);
	}

	/// Clears the vector, removing all values.
	pub fn clear(&mut self) {
		self.inner.clear();
		self.metadata.clear();
	}

	/// Number of items in the `SkVec`.
	///
	/// See [`SkVec` documentation][SkVec] for the distinction between items
	/// and values.
	pub fn len(&self) -> usize {
		self.metadata.len()
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
	fn from(values: Vec<T>) -> Self {
		let mut out = Self::with_capacity(values.len());

		for value in values {
			out.push(value);
		}

		out
	}
}

impl<T: Clone, const N: usize> From<[T; N]> for SkVec<T> {
	fn from(values: [T; N]) -> Self {
		let mut out = Self::with_capacity(values.len());

		for value in values {
			out.push(value);
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
