//! Utility functions for working with 64-bit floats
// TODO: tests

const MANTISSA_MASK: u64 = 0xFFFFFFFFFFFFF;

/// `true` if the sign bit is set (`x` is negative), `false` otherwise
pub fn sign(x: f64) -> bool {
	let bits = x.to_bits();
	(bits >> 63) == 1
}

pub fn exponent_bits(x: f64) -> u16 {
	let bits = x.to_bits();
	let exponent = (bits >> 52) & 0x7FF; // mask clears the sign bit
	exponent.try_into().expect("infallible")
}

/// Returns the true exponent value
///
/// This is the exact value of the exponent, account for bias.  Note that since
/// smoe IEEE floats are normalized (the mantissa has one implicit 52nd bit) the
/// formula `mantissa * 2^true_exponent` won't give the correct value.  You'll
/// need to use [`exponent`], which shifts the value by 52 to account for that.
pub fn true_exponent(x: f64) -> i16 {
	exponent_bits(x) as i16 - 1023 // bias
}

pub fn exponent(x: f64) -> i16 {
	true_exponent(x) - 52
}

pub fn true_mantissa(x: f64) -> u64 {
	x.to_bits() & MANTISSA_MASK
}

pub fn mantissa(x: f64) -> u64 {
	let mut mantissa = true_mantissa(x);
	if exponent_bits(x) == 0 {
		mantissa <<= 1;
	} else {
		mantissa |= 0x10000000000000; // 2^52
	}

	mantissa
}
