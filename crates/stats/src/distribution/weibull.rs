use approx::ulps_eq;

use crate::consts;
use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::statistics::*;
use core::f64;

/// Implements the [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Weibull, Continuous};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let n = Weibull::new(10.0, 1.0).unwrap();
/// assert_almost_eq!(
///     n.mean().unwrap(),
///     0.95135076986687318362924871772654021925505786260884,
///     1e-15
/// );
/// assert_eq!(n.pdf(1.0), 3.6787944117144232159552377016146086744581113103177);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Weibull {
	shape: f64,
	scale: f64,
	scale_pow_shape_inv: f64,
}

/// Represents the errors that can occur when creating a [`Weibull`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum WeibullError {
	/// The shape is NaN, zero or less than zero.
	ShapeInvalid,

	/// The scale is NaN, zero or less than zero.
	ScaleInvalid,
}

impl core::fmt::Display for WeibullError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			WeibullError::ShapeInvalid => write!(
				f,
				"Shape is NaN, zero or less than zero."
			),
			WeibullError::ScaleInvalid => write!(
				f,
				"Scale is NaN, zero or less than zero."
			),
		}
	}
}

impl std::error::Error for WeibullError {}

impl Weibull {
	/// Constructs a new weibull distribution with a shape (k) of `shape`
	/// and a scale (λ) of `scale`
	///
	/// # Errors
	///
	/// Returns an error if `shape` or `scale` are `NaN`.
	/// Returns an error if `shape <= 0.0` or `scale <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Weibull;
	///
	/// let mut result = Weibull::new(10.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = Weibull::new(0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(shape: f64, scale: f64) -> Result<Weibull, WeibullError> {
		if shape.is_nan() || shape <= 0.0 {
			return Err(WeibullError::ShapeInvalid);
		}

		if scale.is_nan() || scale <= 0.0 {
			return Err(WeibullError::ScaleInvalid);
		}

		Ok(Weibull {
			shape,
			scale,
			scale_pow_shape_inv: scale.powf(-shape),
		})
	}

	/// Returns the shape of the weibull distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Weibull;
	///
	/// let n = Weibull::new(10.0, 1.0).unwrap();
	/// assert_eq!(n.shape(), 10.0);
	/// ```
	pub fn shape(&self) -> f64 {
		self.shape
	}

	/// Returns the scale of the weibull distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Weibull;
	///
	/// let n = Weibull::new(10.0, 1.0).unwrap();
	/// assert_eq!(n.scale(), 1.0);
	/// ```
	pub fn scale(&self) -> f64 {
		self.scale
	}
}

impl core::fmt::Display for Weibull {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Weibull({},{})", self.scale, self.shape)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Weibull {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		let x: f64 = rng.random();
		self.scale * (-x.ln()).powf(1.0 / self.shape)
	}
}

impl ContinuousCDF for Weibull {
	/// Calculates the cumulative distribution function for the weibull
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 - e^-((x/λ)^k)
	/// ```
	///
	/// where `k` is the shape and `λ` is the scale
	fn cdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			0.0
		} else {
			-(-x.powf(self.shape) * self.scale_pow_shape_inv)
				.exp_m1()
		}
	}

	/// Calculates the survival function for the weibull
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// e^-((x/λ)^k)
	/// ```
	///
	/// where `k` is the shape and `λ` is the scale
	fn sf(&self, x: f64) -> f64 {
		if x < 0.0 {
			1.0
		} else {
			(-x.powf(self.shape) * self.scale_pow_shape_inv).exp()
		}
	}

	/// Calculates the inverse cumulative distribution function for the weibull
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// λ (-ln(1 - x))^(1 / k)
	/// ```
	///
	/// where `k` is the shape and `λ` is the scale
	fn inverse_cdf(&self, p: f64) -> f64 {
		if !(0.0..=1.0).contains(&p) {
			panic!("x must be in [0, 1]");
		}

		(-((-p).ln_1p() / self.scale_pow_shape_inv))
			.powf(1.0 / self.shape)
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Weibull {
	/// Returns the mean of the weibull distribution
	///
	/// # Formula
	///
	/// ```text
	/// λΓ(1 + 1 / k)
	/// ```
	///
	/// where `k` is the shape, `λ` is the scale, and `Γ` is
	/// the gamma function
	fn mean(&self) -> Option<f64> {
		Some(self.scale * gamma::gamma(1.0 + 1.0 / self.shape))
	}

	/// Returns the median of the weibull distribution
	///
	/// # Formula
	///
	/// ```text
	/// λ(ln(2))^(1 / k)
	/// ```
	///
	/// where `k` is the shape and `λ` is the scale
	fn median(&self) -> Option<f64> {
		Some(self.scale * f64::consts::LN_2.powf(1.0 / self.shape))
	}

	/// Returns the variance of the weibull distribution
	///
	/// # Formula
	///
	/// ```text
	/// λ^2 * (Γ(1 + 2 / k) - Γ(1 + 1 / k)^2)
	/// ```
	///
	/// where `k` is the shape, `λ` is the scale, and `Γ` is
	/// the gamma function
	fn variance(&self) -> Option<f64> {
		let mean = self.mean()?;
		Some(self.scale
			* self.scale * gamma::gamma(1.0 + 2.0 / self.shape)
			- mean * mean)
	}

	/// Returns the entropy of the weibull distribution
	///
	/// # Formula
	///
	/// ```text
	/// γ(1 - 1 / k) + ln(λ / k) + 1
	/// ```
	///
	/// where `k` is the shape, `λ` is the scale, and `γ` is
	/// the Euler-Mascheroni constant
	fn entropy(&self) -> Option<f64> {
		let entr = consts::EULER_MASCHERONI * (1.0 - 1.0 / self.shape)
			+ (self.scale / self.shape).ln()
			+ 1.0;
		Some(entr)
	}

	/// Returns the skewness of the weibull distribution
	///
	/// # Formula
	///
	/// ```text
	/// (Γ(1 + 3 / k) * λ^3 - 3μσ^2 - μ^3) / σ^3
	/// ```
	///
	/// where `k` is the shape, `λ` is the scale, and `Γ` is
	/// the gamma function, `μ` is the mean of the distribution.
	/// and `σ` the standard deviation of the distribution
	fn skewness(&self) -> Option<f64> {
		let mu = self.mean()?;
		let sigma = self.std_dev()?;
		let sigma2 = sigma * sigma;
		let sigma3 = sigma2 * sigma;
		let skew = (self.scale
			* self.scale * self.scale
			* gamma::gamma(1.0 + 3.0 / self.shape)
			- 3.0 * sigma2 * mu - (mu * mu * mu))
			/ sigma3;
		Some(skew)
	}
}

impl Mode<Option<f64>> for Weibull {
	/// Returns the median of the weibull distribution
	///
	/// # Formula
	///
	/// ```text
	/// if k == 1 {
	///     0
	/// } else {
	///     λ((k - 1) / k)^(1 / k)
	/// }
	/// ```
	///
	/// where `k` is the shape and `λ` is the scale
	fn mode(&self) -> Option<f64> {
		let mode = if ulps_eq!(self.shape, 1.0) {
			0.0
		} else {
			self.scale
				* ((self.shape - 1.0) / self.shape)
					.powf(1.0 / self.shape)
		};
		Some(mode)
	}
}

impl Continuous for Weibull {
	type T = f64;

	/// Calculates the probability density function for the weibull
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (k / λ) * (x / λ)^(k - 1) * e^(-(x / λ)^k)
	/// ```
	///
	/// where `k` is the shape and `λ` is the scale
	fn pdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			0.0
		} else if x == 0.0 && ulps_eq!(self.shape, 1.0) {
			1.0 / self.scale
		} else if x.is_infinite() {
			0.0
		} else {
			self.shape
				* (x / self.scale).powf(self.shape - 1.0)
				* (-(x.powf(self.shape))
					* self.scale_pow_shape_inv)
					.exp() / self.scale
		}
	}

	/// Calculates the log probability density function for the weibull
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((k / λ) * (x / λ)^(k - 1) * e^(-(x / λ)^k))
	/// ```
	///
	/// where `k` is the shape and `λ` is the scale
	fn ln_pdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			f64::NEG_INFINITY
		} else if x == 0.0 && ulps_eq!(self.shape, 1.0) {
			0.0 - self.scale.ln()
		} else if x.is_infinite() {
			f64::NEG_INFINITY
		} else {
			self.shape.ln()
				+ (self.shape - 1.0) * (x / self.scale).ln()
				- x.powf(self.shape) * self.scale_pow_shape_inv
				- self.scale.ln()
		}
	}
}
