//! Utility functions for working with 64-bit floats
// TODO: tests

const MANTISSA_MASK: u64 = 0xFFFFFFFFFFFFF;

/// `true` if the sign bit is set (`x` is negative), `false` otherwise
pub fn sign(x: f64) -> bool {
	let bits = x.to_bits();
	(bits >> 63) == 1
}

/// Exponent as raw bits
///
/// This is the unbiased raw value.  To get the mathematical exponent, use
/// [`exponent`].
pub fn exponent_bits(x: f64) -> u16 {
	let bits = x.to_bits();
	let exponent = (bits >> 52) & 0x7FF; // mask clears the sign bit
	exponent.try_into().expect("infallible")
}

/// The mathematical exponent
///
/// This is the value of the exponent.  It's the absolute value of the exponent
/// shifted by 1023, and further by 52 to account for the implicit leading 1 in
/// mantissa.  The value of the float is thus `mantissa * 2^exponent`.
pub fn exponent(x: f64) -> i16 {
	exponent_bits(x) as i16 - 1023 - 52
}

/// Mantissa as raw bits
///
/// This is the raw value.  For the mathematical mantissa use [`mantissa`].
pub fn mantissa_bits(x: f64) -> u64 {
	x.to_bits() & MANTISSA_MASK
}

/// Mathematical mantissa
///
/// It's 53 bits, because denormals are shifted by one and regular values have
/// the implicit one added.
pub fn mantissa(x: f64) -> u64 {
	let mut mantissa = mantissa_bits(x);
	if exponent_bits(x) == 0 {
		// demormals
		mantissa <<= 1;
	} else {
		mantissa |= 0x10000000000000; // 2^52
	}

	mantissa
}
