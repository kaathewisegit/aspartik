//! Provides functions for calculating
//! [harmonic](https://en.wikipedia.org/wiki/Harmonic_number)
//! numbers

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{consts, function::gamma};

/// Computes the `t`-th harmonic number
///
/// Returns `1` as a special case when `t == 0`.
#[cfg_attr(feature = "python", pyfunction)]
pub fn harmonic(t: u64) -> f64 {
	match t {
		0 => 1.0,
		_ => consts::EULER_MASCHERONI + gamma::digamma(t as f64 + 1.0),
	}
}

/// Computes the generalized harmonic number of order `n` of `m`
///
/// Unlike `harmonic`, this function calculated simply as `1 + 1/2^m + 1/3^m +
/// ... + 1/n^m`, so accuracy will degrade for large `m`s and `n`s.
///
/// Returns `1` as a special case when `n == 0`
#[cfg_attr(feature = "python", pyfunction)]
pub fn generalized_harmonic(n: u64, m: f64) -> f64 {
	match n {
		0 => 1.0,
		_ => (0..n).fold(0.0, |acc, x| acc + (x as f64 + 1.0).powf(-m)),
	}
}
