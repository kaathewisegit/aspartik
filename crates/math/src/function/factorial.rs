//! Provides functions related to factorial calculations (e.g. binomial
//! coefficient, factorial, multinomial)

use crate::function::gamma;

/// The maximum factorial representable
/// by a 64-bit floating point without
/// overflowing
pub const MAX_FACTORIAL: usize = 170;

/// Computes the factorial function `x -> x!` for
/// `170 >= x >= 0`. All factorials larger than `170!`
/// will overflow an `f64`.
///
/// # Remarks
///
/// Returns `f64::INFINITY` if `x > 170`
pub fn factorial(x: u64) -> f64 {
	let x = x as usize;
	FCACHE.get(x).map_or(f64::INFINITY, |&fac| fac)
}

/// Computes the logarithmic factorial function `x -> ln(x!)`
/// for `x >= 0`.
///
/// # Remarks
///
/// Returns `0.0` if `x <= 1`
pub fn ln_factorial(x: u64) -> f64 {
	let x = x as usize;
	FCACHE.get(x).map_or_else(
		|| gamma::ln_gamma(x as f64 + 1.0),
		|&fac| fac.ln(),
	)
}

/// Computes the binomial coefficient `n choose k`
/// where `k` and `n` are non-negative values.
///
/// # Remarks
///
/// Returns `0.0` if `k > n`
pub fn binomial(n: u64, k: u64) -> f64 {
	if k > n {
		0.0
	} else {
		(0.5 + (ln_factorial(n)
			- ln_factorial(k) - ln_factorial(n - k))
		.exp())
		.floor()
	}
}

/// Computes the natural logarithm of the binomial coefficient
/// `ln(n choose k)` where `k` and `n` are non-negative values
///
/// # Remarks
///
/// Returns `f64::NEG_INFINITY` if `k > n`
pub fn ln_binomial(n: u64, k: u64) -> f64 {
	if k > n {
		f64::NEG_INFINITY
	} else {
		ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
	}
}

/// Computes the multinomial coefficient: `n choose k_1, k_2, k_3, ...`
///
/// Returns `None` if the elements in `ks` do not sum to `n`.
pub fn multinomial(n: u64, ks: &[u64]) -> Option<f64> {
	let (sum, ret) = ks.iter().fold((0, ln_factorial(n)), |acc, &x| {
		(acc.0 + x, acc.1 - ln_factorial(x))
	});

	if sum == n {
		Some((0.5 + ret.exp()).floor())
	} else {
		None
	}
}

// Initialization for pre-computed cache of 171 factorial
// values 0!...170!
const FCACHE: [f64; MAX_FACTORIAL + 1] = {
	let mut fcache = [1.0; MAX_FACTORIAL + 1];

	// `const` only allow while loops
	let mut i = 1;
	while i < MAX_FACTORIAL + 1 {
		fcache[i] = fcache[i - 1] * i as f64;
		i += 1;
	}

	fcache
};
