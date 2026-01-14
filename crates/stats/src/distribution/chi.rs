use core::f64;
use core::num::NonZeroU64;
use math::function::gamma;

use crate::distribution::{Continuous, ContinuousCDF};
use crate::statistics::*;

/// Implements the [Chi](https://en.wikipedia.org/wiki/Chi_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Chi, Continuous};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let n = Chi::new(2).unwrap();
/// assert_almost_eq!(n.mean().unwrap(), 1.2533141373155003, epsilon = 1e-14);
/// assert_almost_eq!(n.pdf(1.0), 0.6065306597126334, epsilon = 1e-15);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Chi {
	freedom: NonZeroU64,
}

/// Represents the errors that can occur when creating a [`Chi`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum ChiError {
	/// The degrees of freedom are zero.
	FreedomInvalid,
}

impl core::fmt::Display for ChiError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			ChiError::FreedomInvalid => {
				write!(f, "Degrees of freedom are zero")
			}
		}
	}
}

impl std::error::Error for ChiError {}

impl Chi {
	/// Constructs a new chi distribution
	/// with `freedom` degrees of freedom
	///
	/// # Errors
	///
	/// Returns an error if `freedom` is equal to `0`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Chi;
	///
	/// let mut result = Chi::new(2);
	/// assert!(result.is_ok());
	///
	/// result = Chi::new(0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(freedom: u64) -> Result<Chi, ChiError> {
		match NonZeroU64::new(freedom) {
			Some(freedom) => Ok(Self { freedom }),
			None => Err(ChiError::FreedomInvalid),
		}
	}

	/// Returns the degrees of freedom of the chi distribution.
	/// Guaranteed to be non-zero.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Chi;
	///
	/// let n = Chi::new(2).unwrap();
	/// assert_eq!(n.freedom(), 2);
	/// ```
	pub fn freedom(&self) -> u64 {
		self.freedom.get()
	}
}

impl core::fmt::Display for Chi {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "χ_{}", self.freedom)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Chi {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		(0..self.freedom())
			.fold(0.0, |acc, _| {
				acc + super::normal::sample_unchecked(
					rng, 0.0, 1.0,
				)
				.powf(2.0)
			})
			.sqrt()
	}
}

impl ContinuousCDF for Chi {
	/// Calculates the cumulative distribution function for the chi
	/// distribution at `x`.
	///
	/// # Formula
	///
	/// ```text
	/// P(k / 2, x^2 / 2)
	/// ```
	///
	/// where `k` is the degrees of freedom and `P` is
	/// the regularized lower incomplete Gamma function
	fn cdf(&self, x: f64) -> f64 {
		if x == f64::INFINITY {
			1.0
		} else if x <= 0.0 {
			0.0
		} else {
			gamma::gamma_lr(
				self.freedom() as f64 / 2.0,
				x * x / 2.0,
			)
			// `freedom > 0`, so `s > 0`.  `x^2` is either 0 or
			// positive, but `x <= 0` is removed earlir, so it must
			// be positive.  It could overflow, though.
			.unwrap()
		}
	}

	/// `P(k / 2, x^2 / 2)`, where `k` is the degrees of freedom and `P` is
	/// the regularized upper incomplete Gamma function.
	fn sf(&self, x: f64) -> f64 {
		if x == f64::INFINITY {
			0.0
		} else if x <= 0.0 {
			1.0
		} else {
			gamma::gamma_ur(
				self.freedom() as f64 / 2.0,
				x * x / 2.0,
			)
			// Infallible, see `cdf` method
			.unwrap()
		}
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Chi {
	/// Returns the mean of the chi distribution
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// sqrt2 * Γ((k + 1) / 2) / Γ(k / 2)
	/// ```
	///
	/// where `k` is degrees of freedom and `Γ` is the gamma function
	fn mean(&self) -> Option<f64> {
		let freedom = self.freedom() as f64;

		if self.freedom() > 300 {
			// Large n approximation based on the Stirling series approximation to the Gamma function
			// This avoids call the Gamma function with large arguments and returning NaN
			//
			// Relative accuracy follows O(1/n^4) and at 300 d.o.f. is better than 1e-12
			// For a f32 impl the threshold should be changed to 150
			Some((freedom.sqrt())
				/ ((1.0 + 0.25 / freedom)
					* (1.0 + 0.03125
						/ (freedom * freedom)) * (1.0 - 0.046875
					/ (freedom * freedom * freedom))))
		} else {
			let mean = f64::consts::SQRT_2
				* gamma::gamma((freedom + 1.0) / 2.0)
				/ gamma::gamma(freedom / 2.0);
			Some(mean)
		}
	}

	/// Returns the variance of the chi distribution
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// k - μ^2
	/// ```
	///
	/// where `k` is degrees of freedom and `μ` is the mean
	/// of the distribution
	fn variance(&self) -> Option<f64> {
		let mean = self.mean()?;
		Some(self.freedom() as f64 - mean * mean)
	}

	/// Returns the entropy of the chi distribution
	///
	/// # Remarks
	///
	/// Returns `None` if `freedom` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// ln(Γ(k / 2)) + 0.5 * (k - ln2 - (k - 1) * ψ(k / 2))
	/// ```
	///
	/// where `k` is degrees of freedom, `Γ` is the gamma function,
	/// and `ψ` is the digamma function
	fn entropy(&self) -> Option<f64> {
		let freedom = self.freedom() as f64;
		let entr = gamma::ln_gamma(freedom / 2.0)
			+ (freedom
				- (2.0f64).ln() - (freedom - 1.0)
				* gamma::digamma(freedom / 2.0))
				/ 2.0;
		Some(entr)
	}

	/// Returns the skewness of the chi distribution
	///
	/// # Remarks
	///
	/// Returns `NaN` if `freedom` is `INF`
	///
	/// # Formula
	///
	/// ```text
	/// (μ / σ^3) * (1 - 2σ^2)
	/// ```
	/// where `μ` is the mean and `σ` the standard deviation
	/// of the distribution
	fn skewness(&self) -> Option<f64> {
		let sigma = self.std_dev()?;
		let skew = self.mean()? * (1.0 - 2.0 * sigma * sigma)
			/ (sigma * sigma * sigma);
		Some(skew)
	}
}

impl Mode<Option<f64>> for Chi {
	/// Returns the mode for the chi distribution
	///
	/// # Panics
	///
	/// If `freedom < 1.0`
	///
	/// # Formula
	///
	/// ```text
	/// sqrt(k - 1)
	/// ```
	///
	/// where `k` is the degrees of freedom
	fn mode(&self) -> Option<f64> {
		Some(((self.freedom() - 1) as f64).sqrt())
	}
}

impl Continuous for Chi {
	/// Calculates the probability density function for the chi
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2)
	/// ```
	///
	/// where `k` is the degrees of freedom and `Γ` is the gamma function
	fn pdf(&self, x: f64) -> f64 {
		if x == f64::INFINITY || x <= 0.0 {
			0.0
		} else if self.freedom() > 160 {
			self.ln_pdf(x).exp()
		} else {
			let freedom = self.freedom() as f64;
			(2.0f64).powf(1.0 - freedom / 2.0)
				* x.powf(freedom - 1.0) * (-x * x / 2.0).exp()
				/ gamma::gamma(freedom / 2.0)
		}
	}

	/// Calculates the log probability density function for the chi distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2))
	/// ```
	fn ln_pdf(&self, x: f64) -> f64 {
		if x == f64::INFINITY || x <= 0.0 {
			f64::NEG_INFINITY
		} else {
			let freedom = self.freedom() as f64;
			(1.0 - freedom / 2.0) * (2.0f64).ln()
				+ ((freedom - 1.0) * x.ln()) - x * x / 2.0
				- gamma::ln_gamma(freedom / 2.0)
		}
	}
}
