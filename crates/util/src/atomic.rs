//! A module which provides monotonic variants of atomics
//!
//! See the [LLVM reference][m] on monotonic atomics for details.
//!
//! [m]: https://llvm.org/docs/Atomics.html#monotonic

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

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

#[derive(Debug, Default)]
pub struct MonotonicF64(AtomicU64);

impl From<f64> for MonotonicF64 {
	fn from(value: f64) -> Self {
		Self(value.to_bits().into())
	}
}

impl MonotonicF64 {
	pub fn new(value: f64) -> Self {
		value.into()
	}

	pub fn load(&self) -> f64 {
		f64::from_bits(self.0.load(Ordering::Relaxed))
	}

	pub fn store(&self, value: f64) {
		self.0.store(value.to_bits(), Ordering::Relaxed);
	}
}
