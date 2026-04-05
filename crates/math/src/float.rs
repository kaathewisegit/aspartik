//! Utility functions for working with 64-bit floats

#[cfg(feature = "python")]
use pyo3::prelude::*;

const MANTISSA_MASK: u64 = 0xFFFFFFFFFFFFF;

/// `true` if the sign bit is set (`x` is negative), `false` otherwise
#[cfg_attr(feature = "python", pyfunction)]
pub fn sign(x: f64) -> bool {
	let bits = x.to_bits();
	(bits >> 63) == 1
}

/// Exponent as raw bits
///
/// This is the unbiased raw value.  To get the mathematical exponent, use
/// [`exponent`].
#[cfg_attr(feature = "python", pyfunction)]
pub fn exponent_bits(x: f64) -> u16 {
	let bits = x.to_bits();
	let exponent = (bits >> 52) & 0x7FF; // mask clears the sign bit
	exponent.try_into().expect("infallible")
}

/// Mantissa as raw bits
///
/// This is the raw value.  For the mathematical mantissa use [`mantissa`].
#[cfg_attr(feature = "python", pyfunction)]
pub fn mantissa_bits(x: f64) -> u64 {
	x.to_bits() & MANTISSA_MASK
}
