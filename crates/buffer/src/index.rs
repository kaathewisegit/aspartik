use std::{
	cmp::{Eq, Ord},
	fmt::Debug,
	ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

pub trait Size:
	Copy
	+ Add<Output = Self>
	+ AddAssign
	+ Sub<Output = Self>
	+ SubAssign
	+ Mul<Output = Self>
	+ MulAssign
	+ Div<Output = Self>
	+ DivAssign
	+ Debug
	+ Eq
	+ Ord
{
	const ZERO: Self;
	const ONE: Self;
	const TWO: Self;
	const EIGHT: Self;

	fn usize(self) -> usize;

	fn from_usize(idx: usize) -> Option<Self>;
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl Size for u32 {
	const ZERO: u32 = 0;
	const ONE: u32 = 1;
	const TWO: u32 = 2;
	const EIGHT: u32 = 8;

	fn usize(self) -> usize {
		self as usize
	}

	fn from_usize(idx: usize) -> Option<Self> {
		u32::try_from(idx).ok()
	}
}

#[cfg(target_pointer_width = "64")]
impl Size for u64 {
	const ZERO: u64 = 0;
	const ONE: u64 = 1;
	const TWO: u64 = 2;
	const EIGHT: u64 = 8;

	fn usize(self) -> usize {
		self as usize
	}

	fn from_usize(idx: usize) -> Option<Self> {
		u64::try_from(idx).ok()
	}
}

impl Size for usize {
	const ZERO: usize = 0;
	const ONE: usize = 1;
	const TWO: usize = 2;
	const EIGHT: usize = 8;

	fn usize(self) -> usize {
		self
	}

	fn from_usize(idx: usize) -> Option<Self> {
		Some(idx)
	}
}
