//! Provides functions for calculating
//! [harmonic](https://en.wikipedia.org/wiki/Harmonic_number)
//! numbers

use crate::consts;
use crate::function::gamma;

/// Computes the `t`-th harmonic number
///
/// # Remarks
///
/// Returns `1` as a special case when `t == 0`
pub fn harmonic(t: u64) -> f64 {
	match t {
		0 => 1.0,
		_ => consts::EULER_MASCHERONI + gamma::digamma(t as f64 + 1.0),
	}
}

/// Computes the generalized harmonic number of  order `n` of `m`
/// e.g. `(1 + 1/2^m + 1/3^m + ... + 1/n^m)`
///
/// # Remarks
///
/// Returns `1` as a special case when `n == 0`
pub fn gen_harmonic(n: u64, m: f64) -> f64 {
	match n {
		0 => 1.0,
		_ => (0..n).fold(0.0, |acc, x| acc + (x as f64 + 1.0).powf(-m)),
	}
}

#[cfg(test)]
mod tests {
	use crate::assert_almost_eq;

	use super::*;

	#[test]
	fn test_harmonic() {
		assert_eq!(super::harmonic(0), 1.0);

		let cases = [
			(1, 1.0, 1e-14),
			(2, 1.5, 1e-14),
			(4, 2.083333333333333333333, 1e-14),
			(8, 2.717857142857142857143, 1e-14),
			(16, 3.380728993228993228993, 1e-14),
		];
		for (x, expected, epsilon) in cases {
			assert_almost_eq!(harmonic(x), expected, epsilon);
		}
	}

	#[test]
	fn test_gen_harmonic_exact() {
		let cases = [
			((0, 0.0), 1.0),
			((0, f64::INFINITY), 1.0),
			((0, f64::NEG_INFINITY), 1.0),
			((1, 0.0), 1.0),
			((1, f64::INFINITY), 1.0),
			((1, f64::NEG_INFINITY), 1.0),
			((2, 1.0), 1.5),
			((2, 3.0), 1.125),
			((2, f64::INFINITY), 1.0),
			((2, f64::NEG_INFINITY), f64::INFINITY),
			((4, f64::INFINITY), 1.0),
			((4, f64::NEG_INFINITY), f64::INFINITY),
		];

		for ((n, m), expected) in cases {
			assert_eq!(gen_harmonic(n, m), expected);
		}
	}

	#[test]
	fn test_gen_harmonic() {
		let cases = [
			((4, 1.0), 2.083333333333333333333, 1e-14),
			((4, 3.0), 1.177662037037037037037, 1e-16),
		];

		for ((n, m), expected, epsilon) in cases {
			assert_almost_eq!(
				gen_harmonic(n, m),
				expected,
				epsilon
			);
		}
	}
}
