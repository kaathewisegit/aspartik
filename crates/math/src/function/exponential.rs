//! Provides functions related to exponential calculations

use crate::consts;

/// Computes the generalized Exponential Integral function
/// where `x` is the argument and `n` is the integer power of the
/// denominator term.
///
/// Returns `None` if `x < 0.0` or the computation could not
/// converge after 100 iterations
///
/// # Remarks
///
/// This implementation follows the derivation in
///
/// _"Handbook of Mathematical Functions, Applied Mathematics Series, Volume
/// 55"_ - Abramowitz, M., and Stegun, I.A 1964
///
/// AND
///
/// _"Advanced mathematical methods for scientists and engineers"_ - Bender,
/// Carl M.; Steven A. Orszag (1978). page 253
///
/// The continued fraction approach is used for `x > 1.0` while the taylor
/// series expansions is used for `0.0 < x <= 1`.
// TODO: Add examples
pub fn integral(x: f64, n: u64) -> Option<f64> {
	let eps = 0.00000000000000001;
	let max_iter = 100;
	let nf64 = n as f64;
	let near_f64min = 1e-100; // needs very small value that is not quite as small as f64 min

	// special cases
	if n == 0 {
		return Some((-1.0 * x).exp() / x);
	}
	if x == 0.0 {
		return Some(1.0 / (nf64 - 1.0));
	}

	if x > 1.0 {
		let mut b = x + nf64;
		let mut c = 1.0 / near_f64min;
		let mut d = 1.0 / b;
		let mut h = d;
		for i in 1..max_iter + 1 {
			let a = -1.0 * i as f64 * (nf64 - 1.0 + i as f64);
			b += 2.0;
			d = 1.0 / (a * d + b);
			c = b + a / c;
			let del = c * d;
			h *= del;
			if (del - 1.0).abs() < eps {
				return Some(h * (-x).exp());
			}
		}
	} else {
		let mut factorial = 1.0;
		let mut result = if n - 1 != 0 {
			1.0 / (nf64 - 1.0)
		} else {
			-1.0 * x.ln() - consts::EULER_MASCHERONI
		};
		for i in 1..max_iter + 1 {
			factorial *= -1.0 * x / i as f64;
			let del = if i != n - 1 {
				-factorial / (i as f64 - nf64 + 1.0)
			} else {
				let mut psi = -1.0 * consts::EULER_MASCHERONI;
				for ii in 1..n {
					psi += 1.0 / ii as f64;
				}
				factorial * (-1.0 * x.ln() + psi)
			};
			result += del;
			if del.abs() < result.abs() * eps {
				return Some(result);
			}
		}
	}

	None
}
