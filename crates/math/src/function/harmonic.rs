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
