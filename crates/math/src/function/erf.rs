//! Provides the [error](https://en.wikipedia.org/wiki/Error_function) and
//! related functions

#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::f64::consts::PI;

/// Error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erf(x: f64) -> f64 {
	libm::erf(x)
}

/// Complementary error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erfc(x: f64) -> f64 {
	libm::erfc(x)
}

/// Inverse error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erf_inv(x: f64) -> f64 {
	if x == 0.0 {
		0.0
	} else if x >= 1.0 {
		f64::INFINITY
	} else if x <= -1.0 {
		f64::NEG_INFINITY
	} else {
		erf_inv_impl(x)
	}
}

// Uses naive Newton-Raphson method
fn erf_inv_impl(y: f64) -> f64 {
	let mut x = 0.5 * PI.sqrt() * y;

	for _ in 0..20 {
		let error = erf(x) - y;
		let derivative = (2.0 / PI.sqrt()) * (-x.powi(2)).exp();

		x -= error / derivative;
	}

	x
}

/// Complementary inverse error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erfc_inv(x: f64) -> f64 {
	if x <= 0.0 {
		f64::INFINITY
	} else if x >= 2.0 {
		f64::NEG_INFINITY
	} else {
		erfc_inv_impl(x)
	}
}

// Uses Halley's method
fn erfc_inv_impl(y: f64) -> f64 {
	let mut x = if y < 1.0 {
		(-y.ln()).sqrt()
	} else {
		-(-(2.0 - y).ln()).sqrt()
	};

	for _ in 0..20 {
		let f = erfc(x) - y;
		let df = -(2.0 / PI.sqrt()) * (-x.powi(2)).exp();

		// correction
		let numerator = 2.0 * f * df;
		let denominator = 2.0 * (df * df) + 2.0 * x * f * df;

		x -= numerator / denominator;
	}

	x
}
