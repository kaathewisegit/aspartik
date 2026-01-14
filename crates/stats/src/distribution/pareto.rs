use core::f64;

use crate::{
	distribution::{Continuous, ContinuousCDF},
	probability::Probability,
	statistics::{Distribution, Mode},
};

/// Implements the [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Pareto, Continuous};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let p = Pareto::new(1.0, 2.0).unwrap();
/// assert_eq!(p.mean().unwrap(), 2.0);
/// assert_almost_eq!(p.pdf(2.0), 0.25, epsilon = 1e-15);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Pareto {
	scale: f64,
	shape: f64,
}

/// Represents the errors that can occur when creating a [`Pareto`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum ParetoError {
	/// The scale is NaN, zero or less than zero.
	ScaleInvalid,

	/// The shape is NaN, zero or less than zero.
	ShapeInvalid,
}

impl core::fmt::Display for ParetoError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			ParetoError::ScaleInvalid => write!(
				f,
				"Scale is NaN, zero, or less than zero"
			),
			ParetoError::ShapeInvalid => write!(
				f,
				"Shape is NaN, zero, or less than zero"
			),
		}
	}
}

impl std::error::Error for ParetoError {}

impl Pareto {
	/// Constructs a new Pareto distribution with scale `scale`, and `shape`
	/// shape.
	///
	/// # Errors
	///
	/// Returns an error if any of `scale` or `shape` are `NaN`.
	/// Returns an error if `scale <= 0.0` or `shape <= 0.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Pareto;
	///
	/// let mut result = Pareto::new(1.0, 2.0);
	/// assert!(result.is_ok());
	///
	/// result = Pareto::new(0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(scale: f64, shape: f64) -> Result<Pareto, ParetoError> {
		if scale.is_nan() || scale <= 0.0 {
			return Err(ParetoError::ScaleInvalid);
		}

		if shape.is_nan() || shape <= 0.0 {
			return Err(ParetoError::ShapeInvalid);
		}

		Ok(Pareto { scale, shape })
	}

	/// Returns the scale of the Pareto distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Pareto;
	///
	/// let n = Pareto::new(1.0, 2.0).unwrap();
	/// assert_eq!(n.scale(), 1.0);
	/// ```
	pub fn scale(&self) -> f64 {
		self.scale
	}

	/// Returns the shape of the Pareto distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Pareto;
	///
	/// let n = Pareto::new(1.0, 2.0).unwrap();
	/// assert_eq!(n.shape(), 2.0);
	/// ```
	pub fn shape(&self) -> f64 {
		self.shape
	}
}

impl core::fmt::Display for Pareto {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Pareto({},{})", self.scale, self.shape)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Pareto {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		use rand::distr::OpenClosed01;

		// Inverse transform sampling
		let u: f64 = rng.sample(OpenClosed01);
		self.scale * u.powf(-1.0 / self.shape)
	}
}

impl ContinuousCDF for Pareto {
	/// Calculates the cumulative distribution function for the Pareto
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x < x_m {
	///     0
	/// } else {
	///     1 - (x_m/x)^α
	/// }
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn cdf(&self, x: f64) -> f64 {
		if x < self.scale {
			0.0
		} else {
			1.0 - (self.scale / x).powf(self.shape)
		}
	}

	/// Calculates the survival function for the Pareto
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x < x_m {
	///     1
	/// } else {
	///     (x_m/x)^α
	/// }
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn sf(&self, x: f64) -> f64 {
		if x < self.scale {
			1.0
		} else {
			(self.scale / x).powf(self.shape)
		}
	}

	/// `x_m / (1 - x)^(1 / α)`, where `x_m` is the scale and `α` is the
	/// shape.
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		self.scale * (1.0 - *p).powf(-1.0 / self.shape)
	}

	fn lower(&self) -> f64 {
		self.scale
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Pareto {
	/// Returns the mean of the Pareto distribution
	///
	/// # Formula
	///
	/// ```text
	/// if α <= 1 {
	///     f64::INFINITY
	/// } else {
	///     (α * x_m)/(α - 1)
	/// }
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn mean(&self) -> Option<f64> {
		if self.shape <= 1.0 {
			None
		} else {
			Some((self.shape * self.scale) / (self.shape - 1.0))
		}
	}

	/// Returns the median of the Pareto distribution
	///
	/// # Formula
	///
	/// ```text
	/// x_m*2^(1/α)
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn median(&self) -> Option<f64> {
		Some(self.scale * (2f64.powf(1.0 / self.shape)))
	}

	/// Returns the variance of the Pareto distribution
	///
	/// # Formula
	///
	/// ```text
	/// if α <= 2 {
	///     f64::INFINITY
	/// } else {
	///     (x_m/(α - 1))^2 * (α/(α - 2))
	/// }
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn variance(&self) -> Option<f64> {
		if self.shape <= 2.0 {
			None
		} else {
			let a = self.scale / (self.shape - 1.0); // just a temporary variable
			Some(a * a * self.shape / (self.shape - 2.0))
		}
	}

	/// Returns the entropy for the Pareto distribution
	///
	/// # Formula
	///
	/// ```text
	/// ln(α/x_m) - 1/α - 1
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn entropy(&self) -> Option<f64> {
		Some(self.shape.ln()
			- self.scale.ln() - (1.0 / self.shape)
			- 1.0)
	}

	/// Returns the skewness of the Pareto distribution
	///
	/// # Panics
	///
	/// If `α <= 3.0`
	///
	/// where `α` is the shape
	///
	/// # Formula
	///
	/// ```text
	///     (2*(α + 1)/(α - 3))*sqrt((α - 2)/α)
	/// ```
	///
	/// where `α` is the shape
	fn skewness(&self) -> Option<f64> {
		if self.shape <= 3.0 {
			None
		} else {
			Some((2.0 * (self.shape + 1.0) / (self.shape - 3.0))
				* ((self.shape - 2.0) / self.shape).sqrt())
		}
	}
}

impl Mode<Option<f64>> for Pareto {
	/// Returns the mode of the Pareto distribution
	///
	/// # Formula
	///
	/// ```text
	/// x_m
	/// ```
	///
	/// where `x_m` is the scale
	fn mode(&self) -> Option<f64> {
		Some(self.scale)
	}
}

impl Continuous for Pareto {
	/// Calculates the probability density function for the Pareto distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x < x_m {
	///     0
	/// } else {
	///     (α * x_m^α)/(x^(α + 1))
	/// }
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn pdf(&self, x: f64) -> f64 {
		if x < self.scale {
			0.0
		} else {
			(self.shape * self.scale.powf(self.shape))
				/ x.powf(self.shape + 1.0)
		}
	}

	/// Calculates the log probability density function for the Pareto
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x < x_m {
	///     f64::NEG_INFINITY
	/// } else {
	///     ln(α) + α*ln(x_m) - (α + 1)*ln(x)
	/// }
	/// ```
	///
	/// where `x_m` is the scale and `α` is the shape
	fn ln_pdf(&self, x: f64) -> f64 {
		if x < self.scale {
			f64::NEG_INFINITY
		} else {
			self.shape.ln() + self.shape * self.scale.ln()
				- (self.shape + 1.0) * x.ln()
		}
	}
}
