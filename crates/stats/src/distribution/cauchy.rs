use rand::RngExt;
use thiserror::Error;

use core::f64;

use crate::{
	distribution::{Continuous, ContinuousCDF},
	probability::Probability,
	statistics::{Distribution, Mode},
};

/// Implements the [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
/// distribution, also known as the Lorentz distribution.
///
/// # Examples
///
/// ```
/// use stats::distribution::{Cauchy, Continuous};
/// use stats::statistics::Mode;
///
/// let n = Cauchy::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode().unwrap(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.15915494309189535);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Cauchy {
	location: f64,
	scale: f64,
}

/// Represents the errors that can occur when creating a [`Cauchy`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
pub enum CauchyError {
	#[error("The location is NaN")]
	LocationInvalid,

	#[error("The scale is NaN, zero or less than zero")]
	ScaleInvalid,
}

impl Cauchy {
	/// Constructs a new cauchy distribution with the given
	/// location and scale.
	///
	/// # Errors
	///
	/// Returns an error if location or scale are `NaN` or `scale <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Cauchy;
	///
	/// let mut result = Cauchy::new(0.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = Cauchy::new(0.0, -1.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(location: f64, scale: f64) -> Result<Cauchy, CauchyError> {
		if location.is_nan() {
			return Err(CauchyError::LocationInvalid);
		}

		if scale.is_nan() || scale <= 0.0 {
			return Err(CauchyError::ScaleInvalid);
		}

		Ok(Cauchy { location, scale })
	}

	/// Returns the location of the cauchy distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Cauchy;
	///
	/// let n = Cauchy::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.location(), 0.0);
	/// ```
	pub fn location(&self) -> f64 {
		self.location
	}

	/// Returns the scale of the cauchy distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Cauchy;
	///
	/// let n = Cauchy::new(0.0, 1.0).unwrap();
	/// assert_eq!(n.scale(), 1.0);
	/// ```
	pub fn scale(&self) -> f64 {
		self.scale
	}
}

impl core::fmt::Display for Cauchy {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Cauchy({}, {})", self.location, self.scale)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Cauchy {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
		self.location
			+ self.scale
				* (f64::consts::PI * (r.random::<f64>() - 0.5))
					.tan()
	}
}

impl ContinuousCDF for Cauchy {
	/// Calculates the cumulative distribution function for the
	/// cauchy distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / π) * arctan((x - x_0) / γ) + 0.5
	/// ```
	///
	/// where `x_0` is the location and `γ` is the scale
	fn cdf(&self, x: f64) -> f64 {
		(1.0 / f64::consts::PI)
			* ((x - self.location) / self.scale).atan()
			+ 0.5
	}

	/// Calculates the survival function for the
	/// cauchy distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 / π) * arctan(-(x - x_0) / γ) + 0.5
	/// ```
	///
	/// where `x_0` is the location and `γ` is the scale.
	/// note that this is identical to the cdf except for
	/// the negative argument to the arctan function
	fn sf(&self, x: f64) -> f64 {
		(1.0 / f64::consts::PI)
			* ((self.location - x) / self.scale).atan()
			+ 0.5
	}

	/// `x_0 + γ tan((x - 0.5) π)`, where `x_0` is the location and `γ` is
	/// the scale.
	fn inverse_cdf(&self, x: Probability<f64>) -> f64 {
		self.location
			+ self.scale * (f64::consts::PI * (*x - 0.5)).tan()
	}

	fn lower(&self) -> f64 {
		f64::NEG_INFINITY
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Cauchy {
	/// Returns the median of the cauchy distribution
	///
	/// # Formula
	///
	/// ```text
	/// x_0
	/// ```
	///
	/// where `x_0` is the location
	fn median(&self) -> Option<f64> {
		Some(self.location)
	}

	/// Returns the entropy of the cauchy distribution
	///
	/// # Formula
	///
	/// ```text
	/// ln(γ) + ln(4π)
	/// ```
	///
	/// where `γ` is the scale
	fn entropy(&self) -> Option<f64> {
		Some((4.0 * f64::consts::PI * self.scale).ln())
	}
}

impl Mode<Option<f64>> for Cauchy {
	/// Returns the mode of the cauchy distribution
	///
	/// # Formula
	///
	/// ```text
	/// x_0
	/// ```
	///
	/// where `x_0` is the location
	fn mode(&self) -> Option<f64> {
		Some(self.location)
	}
}

impl Continuous for Cauchy {
	/// Calculates the probability density function for the cauchy
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 / (πγ * (1 + ((x - x_0) / γ)^2))
	/// ```
	///
	/// where `x_0` is the location and `γ` is the scale
	fn pdf(&self, x: f64) -> f64 {
		1.0 / (f64::consts::PI
			* self.scale * (1.0
			+ ((x - self.location) / self.scale)
				* ((x - self.location) / self.scale)))
	}

	/// Calculates the log probability density function for the cauchy
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln(1 / (πγ * (1 + ((x - x_0) / γ)^2)))
	/// ```
	///
	/// where `x_0` is the location and `γ` is the scale
	fn ln_pdf(&self, x: f64) -> f64 {
		-(f64::consts::PI
			* self.scale * (1.0
			+ ((x - self.location) / self.scale)
				* ((x - self.location) / self.scale)))
			.ln()
	}
}
