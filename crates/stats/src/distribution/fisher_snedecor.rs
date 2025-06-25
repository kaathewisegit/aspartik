use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::beta;
use crate::statistics::*;

/// Implements the
/// [Fisher-Snedecor](https://en.wikipedia.org/wiki/F-distribution) distribution
/// also commonly known as the F-distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{FisherSnedecor, Continuous};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let n = FisherSnedecor::new(3.0, 3.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 3.0);
/// assert_almost_eq!(n.pdf(1.0), 0.318309886183790671538, 1e-15);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FisherSnedecor {
	freedom_1: f64,
	freedom_2: f64,
}

/// Represents the errors that can occur when creating a [`FisherSnedecor`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum FisherSnedecorError {
	/// `freedom_1` is NaN, infinite, zero or less than zero.
	Freedom1Invalid,

	/// `freedom_2` is NaN, infinite, zero or less than zero.
	Freedom2Invalid,
}

impl core::fmt::Display for FisherSnedecorError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			FisherSnedecorError::Freedom1Invalid => {
				write!(
					f,
					"freedom_1 is NaN, infinite, zero or less than zero."
				)
			}
			FisherSnedecorError::Freedom2Invalid => {
				write!(
					f,
					"freedom_2 is NaN, infinite, zero or less than zero."
				)
			}
		}
	}
}

impl std::error::Error for FisherSnedecorError {}

impl FisherSnedecor {
	/// Constructs a new fisher-snedecor distribution with
	/// degrees of freedom `freedom_1` and `freedom_2`
	///
	/// # Errors
	///
	/// Returns an error if `freedom_1` or `freedom_2` are `NaN`.
	/// Also returns an error if `freedom_1 <= 0.0` or `freedom_2 <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::FisherSnedecor;
	///
	/// let mut result = FisherSnedecor::new(1.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = FisherSnedecor::new(0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(
		freedom_1: f64,
		freedom_2: f64,
	) -> Result<FisherSnedecor, FisherSnedecorError> {
		if !freedom_1.is_finite() || freedom_1 <= 0.0 {
			return Err(FisherSnedecorError::Freedom1Invalid);
		}

		if !freedom_2.is_finite() || freedom_2 <= 0.0 {
			return Err(FisherSnedecorError::Freedom2Invalid);
		}

		Ok(FisherSnedecor {
			freedom_1,
			freedom_2,
		})
	}

	/// Returns the first degree of freedom for the
	/// fisher-snedecor distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::FisherSnedecor;
	///
	/// let n = FisherSnedecor::new(2.0, 3.0).unwrap();
	/// assert_eq!(n.freedom_1(), 2.0);
	/// ```
	pub fn freedom_1(&self) -> f64 {
		self.freedom_1
	}

	/// Returns the second degree of freedom for the
	/// fisher-snedecor distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::FisherSnedecor;
	///
	/// let n = FisherSnedecor::new(2.0, 3.0).unwrap();
	/// assert_eq!(n.freedom_2(), 3.0);
	/// ```
	pub fn freedom_2(&self) -> f64 {
		self.freedom_2
	}
}

impl core::fmt::Display for FisherSnedecor {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "F({},{})", self.freedom_1, self.freedom_2)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for FisherSnedecor {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		(super::gamma::sample_unchecked(rng, self.freedom_1 / 2.0, 0.5)
			* self.freedom_2) / (super::gamma::sample_unchecked(
			rng,
			self.freedom_2 / 2.0,
			0.5,
		) * self.freedom_1)
	}
}

impl ContinuousCDF for FisherSnedecor {
	/// Calculates the cumulative distribution function for the fisher-snedecor
	/// distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// I_((d1 * x) / (d1 * x + d2))(d1 / 2, d2 / 2)
	/// ```
	///
	/// where `d1` is the first degree of freedom, `d2` is
	/// the second degree of freedom, and `I` is the regularized incomplete
	/// beta function
	fn cdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			0.0
		} else if x.is_infinite() {
			1.0
		} else {
			beta::beta_reg(
				self.freedom_1 / 2.0,
				self.freedom_2 / 2.0,
				self.freedom_1 * x
					/ (self.freedom_1 * x + self.freedom_2),
			)
		}
	}

	/// Calculates the survival function for the fisher-snedecor
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// I_(1 - ((d1 * x) / (d1 * x + d2))(d2 / 2, d1 / 2)
	/// ```
	///
	/// where `d1` is the first degree of freedom, `d2` is
	/// the second degree of freedom, and `I` is the regularized incomplete
	/// beta function
	fn sf(&self, x: f64) -> f64 {
		if x < 0.0 {
			1.0
		} else if x.is_infinite() {
			0.0
		} else {
			beta::beta_reg(
				self.freedom_2 / 2.0,
				self.freedom_1 / 2.0,
				1.0 - ((self.freedom_1 * x)
					/ (self.freedom_1 * x
						+ self.freedom_2)),
			)
		}
	}

	/// Calculates the inverse cumulative distribution function for the
	/// fisher-snedecor distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// z = I^{-1}_(x)(d1 / 2, d2 / 2)
	/// d2 / (d1 (1 / z - 1))
	/// ```
	///
	/// where `d1` is the first degree of freedom, `d2` is
	/// the second degree of freedom, and `I` is the regularized incomplete
	/// beta function
	fn inverse_cdf(&self, x: f64) -> f64 {
		if !(0.0..=1.0).contains(&x) {
			panic!("x must be in [0, 1]");
		} else {
			let z = beta::inv_beta_reg(
				self.freedom_1 / 2.0,
				self.freedom_2 / 2.0,
				x,
			);
			self.freedom_2 / (self.freedom_1 * (1.0 / z - 1.0))
		}
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for FisherSnedecor {
	/// Returns the mean of the fisher-snedecor distribution
	///
	/// # Panics
	///
	/// If `freedom_2 <= 2.0`
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom_2` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// d2 / (d2 - 2)
	/// ```
	///
	/// where `d2` is the second degree of freedom
	fn mean(&self) -> Option<f64> {
		if self.freedom_2 <= 2.0 {
			None
		} else {
			Some(self.freedom_2 / (self.freedom_2 - 2.0))
		}
	}

	/// Returns the variance of the fisher-snedecor distribution
	///
	/// # Panics
	///
	/// If `freedom_2 <= 4.0`
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom_1` or `freedom_2` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// (2 * d2^2 * (d1 + d2 - 2)) / (d1 * (d2 - 2)^2 * (d2 - 4))
	/// ```
	///
	/// where `d1` is the first degree of freedom and `d2` is
	/// the second degree of freedom
	fn variance(&self) -> Option<f64> {
		if self.freedom_2 <= 4.0 {
			None
		} else {
			let val = (2.0
				* self.freedom_2 * self.freedom_2
				* (self.freedom_1 + self.freedom_2 - 2.0))
				/ (self.freedom_1
					* (self.freedom_2 - 2.0) * (self.freedom_2
					- 2.0) * (self.freedom_2 - 4.0));
			Some(val)
		}
	}

	/// Returns the skewness of the fisher-snedecor distribution
	///
	/// # Panics
	///
	/// If `freedom_2 <= 6.0`
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom_1` or `freedom_2` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// ((2d1 + d2 - 2) * sqrt(8 * (d2 - 4))) / ((d2 - 6) * sqrt(d1 * (d1 + d2
	/// - 2)))
	/// ```
	///
	/// where `d1` is the first degree of freedom and `d2` is
	/// the second degree of freedom
	fn skewness(&self) -> Option<f64> {
		if self.freedom_2 <= 6.0 {
			None
		} else {
			let val = ((2.0 * self.freedom_1 + self.freedom_2
				- 2.0) * (8.0 * (self.freedom_2 - 4.0))
				.sqrt()) / ((self.freedom_2 - 6.0)
				* (self.freedom_1
					* (self.freedom_1 + self.freedom_2
						- 2.0))
				.sqrt());
			Some(val)
		}
	}
}

impl Mode<Option<f64>> for FisherSnedecor {
	/// Returns the mode for the fisher-snedecor distribution
	///
	/// # Panics
	///
	/// If `freedom_1 <= 2.0`
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom_1` or `freedom_2` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// ((d1 - 2) / d1) * (d2 / (d2 + 2))
	/// ```
	///
	/// where `d1` is the first degree of freedom and `d2` is
	/// the second degree of freedom
	fn mode(&self) -> Option<f64> {
		if self.freedom_1 <= 2.0 {
			None
		} else {
			let val = (self.freedom_2 * (self.freedom_1 - 2.0))
				/ (self.freedom_1 * (self.freedom_2 + 2.0));
			Some(val)
		}
	}
}

impl Continuous for FisherSnedecor {
	type T = f64;

	/// Calculates the probability density function for the fisher-snedecor
	/// distribution
	/// at `x`
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom_1`, `freedom_2` is `INF`, or `x` is `+INF` or
	/// `-INF`
	///
	/// # Formula
	///
	/// ```text
	/// sqrt(((d1 * x) ^ d1 * d2 ^ d2) / (d1 * x + d2) ^ (d1 + d2)) / (x * β(d1
	/// / 2, d2 / 2))
	/// ```
	///
	/// where `d1` is the first degree of freedom, `d2` is
	/// the second degree of freedom, and `β` is the beta function
	fn pdf(&self, x: f64) -> f64 {
		if x.is_infinite() || x <= 0.0 {
			0.0
		} else {
			((self.freedom_1 * x).powf(self.freedom_1)
				* self.freedom_2.powf(self.freedom_2)
				/ (self.freedom_1 * x + self.freedom_2)
					.powf(self.freedom_1 + self.freedom_2))
			.sqrt() / (x * beta::beta(
				self.freedom_1 / 2.0,
				self.freedom_2 / 2.0,
			))
		}
	}

	/// Calculates the log probability density function for the fisher-snedecor
	/// distribution
	/// at `x`
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom_1`, `freedom_2` is `INF`, or `x` is `+INF` or
	/// `-INF`
	///
	/// # Formula
	///
	/// ```text
	/// ln(sqrt(((d1 * x) ^ d1 * d2 ^ d2) / (d1 * x + d2) ^ (d1 + d2)) / (x *
	/// β(d1 / 2, d2 / 2)))
	/// ```
	///
	/// where `d1` is the first degree of freedom, `d2` is
	/// the second degree of freedom, and `β` is the beta function
	fn ln_pdf(&self, x: f64) -> f64 {
		self.pdf(x).ln()
	}
}
