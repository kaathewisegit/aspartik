//! Provides the [gamma](https://en.wikipedia.org/wiki/Gamma_function) and
//! related functions

#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

#[cfg(feature = "python")]
use util::impl_pyerr;

use crate::{
	Positive, consts,
	tolerance::{DEFAULT_F64_ACC, Tolerance},
};

/// Represents the errors that can occur when computing any of the incomplete
/// gamma functions.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(
		skip_from_py_object,
		module = "aspartik.math.functions",
		frozen,
		eq,
		str
	)
)]
pub enum GammaFuncError {
	/// `a` must be a finite non-zero positive number.
	#[error("a is infinite, zero, or negative")]
	AInvalid,

	/// `x` must be a finite non-zero positive number.
	#[error("x is infinite, zero, or negative")]
	XInvalid,
}

#[cfg(feature = "python")]
impl_pyerr!(GammaFuncError, pyo3::exceptions::PyValueError);

/// Auxiliary variable when evaluating the `gamma_ln` function
const GAMMA_R: f64 = 10.900511;

/// Polynomial coefficients for approximating the `gamma_ln` function
const GAMMA_DK: &[f64] = &[
	2.4857408913875355e-5,
	1.0514237858172197,
	-3.4568709722201625,
	4.512277094668948,
	-2.9828522532357664,
	1.056397115771267,
	-1.9542877319164587e-1,
	1.709705434044412e-2,
	-5.719261174043057e-4,
	4.633994733599057e-6,
	-2.7199490848860772e-9,
];

/// Logarithm of the gamma function
///
/// The accuracy should be 16 floating point digits.  The implementation is
/// derived from "An Analysis of the Lanczos Gamma Approximation", Glendon Ralph
/// Pugh, 2004 p. 116
#[cfg_attr(feature = "python", pyfunction)]
pub fn ln_gamma(x: f64) -> f64 {
	if x < 0.5 {
		let s = GAMMA_DK
			.iter()
			.enumerate()
			.skip(1)
			.fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

		consts::LN_PI
			- (consts::PI * x).sin().ln()
			- s.ln() - consts::LN_2_SQRT_E_OVER_PI
			- (0.5 - x) * ((0.5 - x + GAMMA_R) / consts::E).ln()
	} else {
		let s = GAMMA_DK
			.iter()
			.enumerate()
			.skip(1)
			.fold(GAMMA_DK[0], |s, t| {
				s + t.1 / (x + t.0 as f64 - 1.0)
			});

		s.ln() + consts::LN_2_SQRT_E_OVER_PI
			+ (x - 0.5) * ((x - 0.5 + GAMMA_R) / consts::E).ln()
	}
}

/// Gamma function
///
/// The accuracy should be 16 floating point digits.  The implementation is
/// derived from "An Analysis of the Lanczos Gamma Approximation", Glendon Ralph
/// Pugh, 2004 p. 116
#[cfg_attr(feature = "python", pyfunction)]
pub fn gamma(x: f64) -> f64 {
	if x < 0.5 {
		let s = GAMMA_DK
			.iter()
			.enumerate()
			.skip(1)
			.fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

		consts::PI
			/ ((consts::PI * x).sin()
				* s * consts::TWO_SQRT_E_OVER_PI
				* ((0.5 - x + GAMMA_R) / consts::E)
					.powf(0.5 - x))
	} else {
		let s = GAMMA_DK
			.iter()
			.enumerate()
			.skip(1)
			.fold(GAMMA_DK[0], |s, t| {
				s + t.1 / (x + t.0 as f64 - 1.0)
			});

		s * consts::TWO_SQRT_E_OVER_PI
			* ((x - 0.5 + GAMMA_R) / consts::E).powf(x - 0.5)
	}
}

/// Upper incomplete gamma function.
///
/// The formula is `Γ(s,x) = ∫_x^inf t^(s-1) exp(-t)` for `s > 0` and `x > 0`,
/// where `a` is the argument for the gamma function and `x` is the lower
/// intergral limit.
#[cfg_attr(feature = "python", pyfunction)]
pub fn gamma_ui(s: Positive<f64>, x: Positive<f64>) -> f64 {
	gamma_ur(s, x) * gamma(*s)
}

/// Lower incomplete gamma function.
///
/// The formula is `Γ(s,x) = ∫_0^s t^(s-1) exp(-t)` for `s > 0` and `x > 0`,
/// where `a` is the argument for the gamma function and `x` is the lower
/// intergral limit.
#[cfg_attr(feature = "python", pyfunction)]
pub fn gamma_li(s: Positive<f64>, x: Positive<f64>) -> f64 {
	gamma_lr(s, x) * gamma(*s)
}

/// Upper incomplete regularized gamma function
///
/// The formula is `gamma_ui(s, x) / gamma(s)`.
#[cfg_attr(feature = "python", pyfunction)]
pub fn gamma_ur(s: Positive<f64>, x: Positive<f64>) -> f64 {
	let (so, xo) = (s, x);
	let (s, x) = (*s, *x);

	let eps = 0.000000000000001;
	let big = 4503599627370496.0;
	let big_inv = 2.220446049250313e-16;

	if x < 1.0 || x <= s {
		return 1.0 - gamma_lr(so, xo);
	}

	let mut ax = s * x.ln() - x - ln_gamma(s);
	if ax < -709.782712893384 {
		return if s < x { 0.0 } else { 1.0 };
	}

	ax = ax.exp();
	let mut y = 1.0 - s;
	let mut z = x + y + 1.0;
	let mut c = 0.0;
	let mut pkm2 = 1.0;
	let mut qkm2 = x;
	let mut pkm1 = x + 1.0;
	let mut qkm1 = z * x;
	let mut ans = pkm1 / qkm1;
	loop {
		y += 1.0;
		z += 2.0;
		c += 1.0;
		let yc = y * c;
		let pk = pkm1 * z - pkm2 * yc;
		let qk = qkm1 * z - qkm2 * yc;

		pkm2 = pkm1;
		pkm1 = pk;
		qkm2 = qkm1;
		qkm1 = qk;

		if pk.abs() > big {
			pkm2 *= big_inv;
			pkm1 *= big_inv;
			qkm2 *= big_inv;
			qkm1 *= big_inv;
		}

		if qk != 0.0 {
			let r = pk / qk;
			let t = ((ans - r) / r).abs();
			ans = r;

			if t <= eps {
				break;
			}
		}
	}
	ans * ax
}

/// Lower incomplete regularized gamma function
///
/// The formula is `gamma_li(s, x) / gamma(s)`.
#[cfg_attr(feature = "python", pyfunction)]
pub fn gamma_lr(s: Positive<f64>, x: Positive<f64>) -> f64 {
	let (s, x) = (*s, *x);

	let eps = 0.000000000000001;
	let big = 4503599627370496.0;
	let big_inv = 2.220446049250313e-16;

	if s.abs_diff(&0.0) <= DEFAULT_F64_ACC {
		return 1.0;
	}
	if x.abs_diff(&0.0) <= DEFAULT_F64_ACC {
		return 0.0;
	}

	let ax = s * x.ln() - x - ln_gamma(s);
	if ax < -709.782712893384 {
		if s < x {
			return 1.0;
		}
		return 0.0;
	}
	if x <= 1.0 || x <= s {
		let mut r2 = s;
		let mut c2 = 1.0;
		let mut ans2 = 1.0;
		loop {
			r2 += 1.0;
			c2 *= x / r2;
			ans2 += c2;

			if c2 / ans2 <= eps {
				break;
			}
		}
		return ax.exp() * ans2 / s;
	}

	let mut y = 1.0 - s;
	let mut z = x + y + 1.0;
	let mut c = 0;

	let mut p3 = 1.0;
	let mut q3 = x;
	let mut p2 = x + 1.0;
	let mut q2 = z * x;
	let mut ans = p2 / q2;

	loop {
		y += 1.0;
		z += 2.0;
		c += 1;
		let yc = y * f64::from(c);

		let p = p2 * z - p3 * yc;
		let q = q2 * z - q3 * yc;

		p3 = p2;
		p2 = p;
		q3 = q2;
		q2 = q;

		if p.abs() > big {
			p3 *= big_inv;
			p2 *= big_inv;
			q3 *= big_inv;
			q2 *= big_inv;
		}

		if q != 0.0 {
			let nextans = p / q;
			let error = ((ans - nextans) / nextans).abs();
			ans = nextans;

			if error <= eps {
				break;
			}
		}
	}
	1.0 - ax.exp() * ans
}

/// Digamma function
///
/// Digamma is defined as the derivative of the log of the gamma function.  The
/// implementation is based on "Algorithm AS 103", Jose Bernardo, Applied
/// Statistics, Volume 25, Number 3 1976, pages 315 - 317.
#[cfg_attr(feature = "python", pyfunction)]
pub fn digamma(x: f64) -> f64 {
	let c = 12.0;
	let d1 = -0.5772156649015329;
	let d2 = 1.6449340668482264;
	let s = 1e-6;
	let s3 = 1.0 / 12.0;
	let s4 = 1.0 / 120.0;
	let s5 = 1.0 / 252.0;
	let s6 = 1.0 / 240.0;
	let s7 = 1.0 / 132.0;

	if x == f64::NEG_INFINITY || x.is_nan() {
		return f64::NAN;
	}
	if x <= 0.0 && x.floor() == x {
		return f64::NEG_INFINITY;
	}
	if x < 0.0 {
		return digamma(1.0 - x) + consts::PI / (-consts::PI * x).tan();
	}
	if x <= s {
		return d1 - 1.0 / x + d2 * x;
	}

	let mut result = 0.0;
	let mut z = x;
	while z < c {
		result -= 1.0 / z;
		z += 1.0;
	}

	if z >= c {
		let mut r = 1.0 / z;
		result += z.ln() - 0.5 * r;
		r *= r;

		result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
	}
	result
}

/// Inverse digamma function
///
/// This function propagates NaNs.
#[cfg_attr(feature = "python", pyfunction)]
pub fn digamma_inv(x: f64) -> f64 {
	if x.is_nan() {
		return f64::NAN;
	}
	if x == f64::NEG_INFINITY {
		return 0.0;
	}
	if x == f64::INFINITY {
		return f64::INFINITY;
	}
	let mut y = x.exp();
	let mut i = 1.0;
	while i > 1e-15 {
		y += i * signum(x - digamma(y));
		i /= 2.0;
	}
	y
}

/// Modified signum that returns 0.0 if x == 0.0
// XXX: used by inv_digamma, consider extracting into a public method
fn signum(x: f64) -> f64 {
	if x == 0.0 { 0.0 } else { x.signum() }
}
