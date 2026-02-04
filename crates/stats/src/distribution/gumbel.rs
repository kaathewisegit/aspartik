use rand::RngExt;
use thiserror::Error;

use core::f64::consts::PI;

use super::{Continuous, ContinuousCDF};
use crate::{
	probability::Probability,
	statistics::{Distribution, Mode},
};
use math::consts::EULER_MASCHERONI;

/// Implements the [Gumbel](https://en.wikipedia.org/wiki/Gumbel_distribution)
/// distribution, also known as the type-I generalized extreme value distribution.
///
/// # Examples
///
/// ```
/// use stats::distribution::{Gumbel, Continuous};
/// use stats::statistics::Distribution;
///
/// let n = Gumbel::new(0.0, 1.0).unwrap();
/// assert_eq!(n.location(), 0.0);
/// assert_eq!(n.skewness().unwrap(), 1.13955);
/// assert_eq!(n.pdf(0.0), 0.36787944117144233);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Gumbel {
	location: f64,
	scale: f64,
}

/// Represents the errors that can occur when creating a [`Gumbel`]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
pub enum GumbelError {
	#[error("The location is NaN")]
	LocationInvalid,

	#[error("The scale is NaN, zero or less than zero")]
	ScaleInvalid,
}

impl Gumbel {
	/// Constructs a new Gumbel distribution with the given
	/// location and scale.
	///
	/// # Errors
	///
	/// Returns an error if location or scale are `NaN` or `scale <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Gumbel;
	///
	/// let mut result = Gumbel::new(0.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = Gumbel::new(0.0, -1.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(location: f64, scale: f64) -> Result<Self, GumbelError> {
		if location.is_nan() {
			return Err(GumbelError::LocationInvalid);
		}

		if scale.is_nan() || scale <= 0.0 {
			return Err(GumbelError::ScaleInvalid);
		}

		Ok(Self { location, scale })
	}

	/// Returns the location of the Gumbel distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Gumbel;
	///
	/// let n = Gumbel::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.location(), 0.0);
	/// ```
	pub fn location(&self) -> f64 {
		self.location
	}

	/// Returns the scale of the Gumbel distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Gumbel;
	///
	/// let n = Gumbel::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.scale(), 1.0);
	/// ```
	pub fn scale(&self) -> f64 {
		self.scale
	}
}

impl core::fmt::Display for Gumbel {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		write!(f, "Gumbel({:?}, {:?})", self.location, self.scale)
	}
}

#[cfg(feature = "rand")]
impl rand::distr::Distribution<f64> for Gumbel {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
		self.location - self.scale * ((-(r.random::<f64>())).ln()).ln()
	}
}

impl ContinuousCDF for Gumbel {
	/// Calculates the cumulative distribution function for the
	/// Gumbel distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// e^(-e^(-(x - μ) / β))
	/// ```
	///
	/// where `μ` is the location and `β` is the scale
	fn cdf(&self, x: f64) -> f64 {
		(-(-(x - self.location) / self.scale).exp()).exp()
	}

	/// `μ - β ln(-ln(p))` for 0 < p < 1, where `μ` is the location and `β`
	/// is the scale, and infinities at the ends.
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		let p = *p;

		if p == 0.0 {
			f64::NEG_INFINITY
		} else if p == 1.0 {
			f64::INFINITY
		} else {
			self.location - self.scale * ((-(p.ln())).ln())
		}
	}

	/// Calculates the survival function for the
	/// Gumbel distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 - e^(-e^(-(x - μ) / β))
	/// ```
	///
	/// where `μ` is the location and `β` is the scale
	fn sf(&self, x: f64) -> f64 {
		-(-(-(x - self.location) / self.scale).exp()).exp_m1()
	}

	fn lower(&self) -> f64 {
		f64::NEG_INFINITY
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Gumbel {
	/// Returns the mean of the Gumbel distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ + γβ
	/// ```
	///
	/// where `μ` is the location, `β` is the scale
	/// and `γ` is the Euler-Mascheroni constant (approx 0.57721)
	fn mean(&self) -> Option<f64> {
		Some(self.location + (EULER_MASCHERONI * self.scale))
	}

	/// Returns the median of the Gumbel distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ - β ln(ln(2))
	/// ```
	///
	/// where `μ` is the location and `β` is the scale parameter
	fn median(&self) -> Option<f64> {
		Some(self.location - self.scale * (((2.0_f64).ln()).ln()))
	}

	/// Returns the variance of the Gumbel distribution
	///
	/// # Formula
	///
	/// ```text
	/// (π^2 / 6) * β^2
	/// ```
	///
	/// where `β` is the scale and `π` is the constant PI (approx 3.14159)
	fn variance(&self) -> Option<f64> {
		Some(((PI * PI) / 6.0) * self.scale * self.scale)
	}

	/// Returns the standard deviation of the Gumbel distribution
	///
	/// # Formula
	///
	/// ```text
	/// β * π / sqrt(6)
	/// ```
	///
	/// where `β` is the scale and `π` is the constant PI (approx 3.14159)
	fn std_dev(&self) -> Option<f64> {
		Some(self.scale * PI / 6.0_f64.sqrt())
	}

	/// Returns the entropy of the Gumbel distribution
	///
	/// # Formula
	///
	/// ```text
	/// ln(β) + γ + 1
	/// ```
	///
	/// where `β` is the scale
	/// and `γ` is the Euler-Mascheroni constant (approx 0.57721)
	fn entropy(&self) -> Option<f64> {
		Some(1.0 + EULER_MASCHERONI + (self.scale).ln())
	}

	/// Returns the skewness of the Gumbel distribution
	///
	/// # Formula
	///
	/// ```text
	/// 12 * sqrt(6) * ζ(3) / π^3 ≈ 1.13955
	/// ```
	/// ζ(3) is the Riemann zeta function evaluated at 3 (approx 1.20206)
	/// and π is the constant PI (approx 3.14159)
	///
	/// This approximately evaluates to 1.13955
	fn skewness(&self) -> Option<f64> {
		Some(1.13955)
	}
}

impl Mode<f64> for Gumbel {
	/// Returns the mode of the Gumbel distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the location
	fn mode(&self) -> f64 {
		self.location
	}
}

impl Continuous for Gumbel {
	/// Calculates the probability density function for the Gumbel
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1/β) * exp(-(x - μ)/β) * exp(-exp(-(x - μ)/β))
	/// ```
	///
	/// where `μ` is the location, `β` is the scale
	fn pdf(&self, x: f64) -> f64 {
		(1.0_f64 / self.scale)
			* (-(x - self.location) / (self.scale)).exp()
			* (-((-(x - self.location) / self.scale).exp())).exp()
	}

	/// Calculates the log probability density function for the Gumbel
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((1/β) * exp(-(x - μ)/β) * exp(-exp(-(x - μ)/β)))
	/// ```
	///
	/// where `μ` is the location, `β` is the scale
	fn ln_pdf(&self, x: f64) -> f64 {
		((1.0_f64 / self.scale)
			* (-(x - self.location) / (self.scale)).exp()
			* (-((-(x - self.location) / self.scale).exp())).exp())
		.ln()
	}
}
