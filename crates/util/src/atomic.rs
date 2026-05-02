//! A module which provides monotonic variants of atomics
//!
//! See the [LLVM reference][m] on monotonic atomics for details.
//!
//! [m]: https://llvm.org/docs/Atomics.html#monotonic

use std::sync::atomic::{AtomicU32, Ordering};

/// An atomic `u32` on which all operations are relaxed
#[derive(Debug, Default)]
pub struct MonotonicU32(AtomicU32);

impl From<u32> for MonotonicU32 {
	fn from(value: u32) -> Self {
		Self(value.into())
	}
}

impl MonotonicU32 {
	pub fn new(value: u32) -> Self {
		value.into()
	}

	pub fn load(&self) -> u32 {
		self.0.load(Ordering::Relaxed)
	}

	pub fn store(&self, value: u32) {
		self.0.store(value, Ordering::Relaxed)
	}

	pub fn add(&self, value: u32) {
		self.0.fetch_add(value, Ordering::Relaxed);
	}
}
