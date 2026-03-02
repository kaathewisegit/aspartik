//! Primitive bitmap implementation
//!
//! It's used by `Tree` in b3 to track updated nodes, so it only needs to track
//! a fixed amount of bits after it is created.  This means I don't need roaring
//! bitmaps.  And when I used [BitVec] with SkVec, I found it to be too focused
//! on indexing, which itself was very slow.
//!
//! [BitVec]: https://lib.rs/crates/bitvec

use serde::{Deserialize, Serialize};

/// A basic contiguous byte slice bitmap
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bitmap {
	inner: Box<[u8]>,
}

impl Bitmap {
	/// Create a new bitmap with capacity of `size`
	///
	/// `size` will be rounded up to the next multiple of 8.  These
	/// additional bits will be set to false and are flipped in
	/// [`set_all_on`] and accounted for in [`is_any_on`]
	///
	/// [`set_all_on`]: Self::set_all_on
	/// [`is_any_on`]: Self::is_any_on
	pub fn new(size: usize) -> Self {
		let length = size.div_ceil(8);
		Bitmap {
			inner: (0..length).map(|_| 0).collect(),
		}
	}

	/// Set the `index`th bit to 1 if `on` is true or 0 if `on` is false
	pub fn set(&mut self, index: usize, on: bool) {
		if on {
			self.set_on(index)
		} else {
			self.set_off(index)
		}
	}

	/// Set `index`th bit to 1
	pub fn set_on(&mut self, index: usize) {
		let byte_index = index / 8;
		let bit_value = 1 << (index % 8);
		self.inner[byte_index] |= bit_value;
	}

	/// Set `index`th bit to 0
	pub fn set_off(&mut self, index: usize) {
		let byte_index = index / 8;
		let mask = !(1 << (index % 8));
		self.inner[byte_index] &= mask;
	}

	/// Returns `true` if the `index`th bit is 1 and `false` if it is 0
	// `inline` because `Bitmap` is not generic and this is a separate
	// crate, so that's a rare case where `inline` is needed.  I only
	// realized it after seeing this method on the flamegraph.
	#[inline]
	pub fn at(&self, index: usize) -> bool {
		let byte_index = index / 8;
		let bit_value = 1 << (index % 8);
		(self.inner[byte_index] & bit_value) != 0
	}

	/// Clears all bits
	pub fn set_all_off(&mut self) {
		for byte in &mut self.inner {
			*byte = 0;
		}
	}

	/// Sets all bits to 1
	///
	/// This includes the padding bits (see [`new`]).
	///
	/// [`new`]: Self::new
	pub fn set_all_on(&mut self) {
		for byte in &mut self.inner {
			*byte = 0b1111_1111;
		}
	}

	/// Returns `true` if any of the bits is 1
	///
	/// This includes the padding bits.
	pub fn is_any_on(&self) -> bool {
		self.inner.iter().any(|&b| b != 0)
	}
}
