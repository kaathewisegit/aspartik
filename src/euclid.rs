//! Provides number theory utility functions

/// Provides a trait for the canonical modulus operation since % is technically
/// the remainder operation
pub trait Modulus {
	/// Performs a canonical modulus operation between `self` and `divisor`.
	///
	/// # Examples
	///
	/// ```
	/// use statrs::euclid::Modulus;
	///
	/// let x = 4i64.modulus(5);
	/// assert_eq!(x, 4);
	///
	/// let y = -4i64.modulus(5);
	/// assert_eq!(x, 4);
	/// ```
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
