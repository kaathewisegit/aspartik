//! Provides number theory utility functions

/// The canonical modulus operation.
///
/// The Rust built-in `Rem` operator (`%`) calculates a remainder, which
/// preserves the sign of the left hand argument (see the [std docs]).  This
/// trait provides a method which always returns a positive a positive
/// remainder, as per the mathematical modulo definition.
///
/// [std docs]: https://doc.rust-lang.org/stable/std/ops/trait.Rem.html#impl-Rem-for-i8
pub trait Modulus {
	/// Performs a canonical modulus operation between `self` and `divisor`.
	///
	/// This method is derived automatically for all types which implement
	/// `Add` and `Rem` using the `((self % divisor) + divisor) % divisor`
	/// formula.
	fn modulus(self, divisor: Self) -> Self;
}

use core::ops::{Add, Rem};

impl<T> Modulus for T
where
	T: Add<Output = T> + Rem<Output = T> + Copy,
{
	fn modulus(self, divisor: T) -> T {
		((self % divisor) + divisor) % divisor
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn basic() {
		assert_eq!(4, 4i64.modulus(5));
		assert_eq!(1, (-4i64).modulus(5));
	}
}
