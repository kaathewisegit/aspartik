use core::f64;

use crate::{
	distribution::{Continuous, ContinuousCDF},
	probability::Probability,
	statistics::{Distribution, Mode},
};

/// Implements the
/// [Triangular](https://en.wikipedia.org/wiki/Triangular_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Triangular, Continuous};
/// use stats::statistics::Distribution;
///
/// let n = Triangular::new(0.0, 5.0, 2.5).unwrap();
/// assert_eq!(n.mean().unwrap(), 7.5 / 3.0);
/// assert_eq!(n.pdf(2.5), 5.0 / 12.5);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Triangular {
	min: f64,
	max: f64,
	mode: f64,
}

/// Represents the errors that can occur when creating a [`Triangular`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum TriangularError {
	/// The minimum is NaN or infinite.
	MinInvalid,

	/// The maximum is NaN or infinite.
	MaxInvalid,

	/// The mode is NaN or infinite.
	ModeInvalid,

	/// The mode is less than the minimum or greater than the maximum.
	ModeOutOfRange,

	/// The minimum equals the maximum.
	MinEqualsMax,
}

impl core::fmt::Display for TriangularError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			TriangularError::MinInvalid => {
				write!(f, "Minimum is NaN or infinite.")
			}
			TriangularError::MaxInvalid => {
				write!(f, "Maximum is NaN or infinite.")
			}
			TriangularError::ModeInvalid => {
				write!(f, "Mode is NaN or infinite.")
			}
			TriangularError::ModeOutOfRange => {
				write!(
					f,
					"Mode is less than minimum or greater than maximum"
				)
			}
			TriangularError::MinEqualsMax => {
				write!(f, "Minimum equals Maximum")
			}
		}
	}
}

impl std::error::Error for TriangularError {}

impl Triangular {
	/// Constructs a new triangular distribution with a minimum of `min`,
	/// maximum of `max`, and a mode of `mode`.
	///
	/// # Errors
	///
	/// Returns an error if `min`, `max`, or `mode` are `NaN` or `±INF`.
	/// Returns an error if `max < mode`, `mode < min`, or `max == min`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Triangular;
	///
	/// let mut result = Triangular::new(0.0, 5.0, 2.5);
	/// assert!(result.is_ok());
	///
	/// result = Triangular::new(2.5, 1.5, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(
		min: f64,
		max: f64,
		mode: f64,
	) -> Result<Triangular, TriangularError> {
		if !min.is_finite() {
			return Err(TriangularError::MinInvalid);
		}

		if !max.is_finite() {
			return Err(TriangularError::MaxInvalid);
		}

		if !mode.is_finite() {
			return Err(TriangularError::ModeInvalid);
		}

		if max < mode || mode < min {
			return Err(TriangularError::ModeOutOfRange);
		}

		if min == max {
			return Err(TriangularError::MinEqualsMax);
		}

		Ok(Triangular { min, max, mode })
	}

	/// Returns the minimum value in the domain of the
	/// triangular distribution
	///
	/// # Remarks
	///
	/// The return value is the same min used to construct the distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Triangular;
	///
	/// let n = Triangular::new(0.0, 5.0, 2.5).unwrap();
	/// assert_eq!(n.min(), 0.0);
	/// ```
	pub fn min(&self) -> f64 {
		self.min
	}

	/// Returns the maximum value in the domain of the
	/// triangular distribution
	///
	/// # Remarks
	///
	/// The return value is the same max used to construct the distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Triangular;
	///
	/// let n = Triangular::new(0.0, 5.0, 2.5).unwrap();
	/// assert_eq!(n.max(), 5.0);
	/// ```
	pub fn max(&self) -> f64 {
		self.max
	}
}

impl core::fmt::Display for Triangular {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(
			f,
			"Triangular([{},{}], {})",
			self.min, self.max, self.mode
		)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Triangular {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		sample_unchecked(rng, self.min, self.max, self.mode)
	}
}

impl ContinuousCDF for Triangular {
	/// Calculates the cumulative distribution function for the triangular
	/// distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x == min {
	///     0
	/// } if min < x <= mode {
	///     (x - min)^2 / ((max - min) * (mode - min))
	/// } else if mode < x < max {
	///     1 - (max - x)^2 / ((max - min) * (max - mode))
	/// } else {
	///     1
	/// }
	/// ```
	fn cdf(&self, x: f64) -> f64 {
		let a = self.min;
		let b = self.max;
		let c = self.mode;
		if x <= a {
			0.0
		} else if x <= c {
			(x - a) * (x - a) / ((b - a) * (c - a))
		} else if x < b {
			1.0 - (b - x) * (b - x) / ((b - a) * (b - c))
		} else {
			1.0
		}
	}

	/// Calculates the survival function for the triangular
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x == min {
	///     1
	/// } if min < x <= mode {
	///     1 - (x - min)^2 / ((max - min) * (mode - min))
	/// } else if mode < x < max {
	///     (max - min)^2 / ((max - min) * (max - mode))
	/// } else {
	///     0
	/// }
	/// ```
	fn sf(&self, x: f64) -> f64 {
		let a = self.min;
		let b = self.max;
		let c = self.mode;
		if x <= a {
			1.0
		} else if x <= c {
			1.0 - ((x - a) * (x - a) / ((b - a) * (c - a)))
		} else if x < b {
			(b - x) * (b - x) / ((b - a) * (b - c))
		} else {
			0.0
		}
	}

	/// # Formula
	///
	/// ```text
	/// if x < (mode - min) / (max - min) {
	///     min + ((max - min) * (mode - min) * x)^(1 / 2)
	/// } else {
	///     max - ((max - min) * (max - mode) * (1 - x))^(1 / 2)
	/// }
	/// ```
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		let min = self.min;
		let max = self.max;
		let mode = self.mode;
		let p = *p;

		if p < (mode - min) / (max - min) {
			min + ((mode - min) * (max - min) * p).sqrt()
		} else {
			max - ((max - min) * (max - mode) * (1.0 - p)).sqrt()
		}
	}

	fn lower(&self) -> f64 {
		self.min
	}

	fn upper(&self) -> f64 {
		self.max
	}
}

impl Distribution for Triangular {
	/// Returns the mean of the triangular distribution
	///
	/// # Formula
	///
	/// ```text
	/// (min + max + mode) / 3
	/// ```
	fn mean(&self) -> Option<f64> {
		Some((self.min + self.max + self.mode) / 3.0)
	}

	/// Returns the median of the triangular distribution
	///
	/// # Formula
	///
	/// ```text
	/// if mode >= (min + max) / 2 {
	///     min + sqrt((max - min) * (mode - min) / 2)
	/// } else {
	///     max - sqrt((max - min) * (max - mode) / 2)
	/// }
	/// ```
	fn median(&self) -> Option<f64> {
		let a = self.min;
		let b = self.max;
		let c = self.mode;
		let median = if c >= (a + b) / 2.0 {
			a + ((b - a) * (c - a) / 2.0).sqrt()
		} else {
			b - ((b - a) * (b - c) / 2.0).sqrt()
		};
		Some(median)
	}

	/// Returns the variance of the triangular distribution
	///
	/// # Formula
	///
	/// ```text
	/// (min^2 + max^2 + mode^2 - min * max - min * mode - max * mode) / 18
	/// ```
	fn variance(&self) -> Option<f64> {
		let a = self.min;
		let b = self.max;
		let c = self.mode;
		Some((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0)
	}

	/// Returns the entropy of the triangular distribution
	///
	/// # Formula
	///
	/// ```text
	/// 1 / 2 + ln((max - min) / 2)
	/// ```
	fn entropy(&self) -> Option<f64> {
		Some(0.5 + ((self.max - self.min) / 2.0).ln())
	}

	/// Returns the skewness of the triangular distribution
	///
	/// # Formula
	///
	/// ```text
	/// (sqrt(2) * (min + max - 2 * mode) * (2 * min - max - mode) * (min - 2 *
	/// max + mode)) /
	/// ( 5 * (min^2 + max^2 + mode^2 - min * max - min * mode - max * mode)^(3
	/// / 2))
	/// ```
	fn skewness(&self) -> Option<f64> {
		let a = self.min;
		let b = self.max;
		let c = self.mode;
		let q = f64::consts::SQRT_2
			* (a + b - 2.0 * c) * (2.0 * a - b - c)
			* (a - 2.0 * b + c);
		let d = 5.0
			* (a * a + b * b + c * c - a * b - a * c - b * c)
				.powf(3.0 / 2.0);
		Some(q / d)
	}
}

impl Mode<Option<f64>> for Triangular {
	/// Returns the mode of the triangular distribution
	///
	/// # Formula
	///
	/// ```text
	/// mode
	/// ```
	fn mode(&self) -> Option<f64> {
		Some(self.mode)
	}
}

impl Continuous for Triangular {
	/// Calculates the probability density function for the triangular
	/// distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x < min {
	///     0
	/// } else if min <= x <= mode {
	///     2 * (x - min) / ((max - min) * (mode - min))
	/// } else if mode < x <= max {
	///     2 * (max - x) / ((max - min) * (max - mode))
	/// } else {
	///     0
	/// }
	/// ```
	fn pdf(&self, x: f64) -> f64 {
		let a = self.min;
		let b = self.max;
		let c = self.mode;
		if a <= x && x <= c {
			2.0 * (x - a) / ((b - a) * (c - a))
		} else if c < x && x <= b {
			2.0 * (b - x) / ((b - a) * (b - c))
		} else {
			0.0
		}
	}

	/// Calculates the log probability density function for the triangular
	/// distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln( if x < min {
	///     0
	/// } else if min <= x <= mode {
	///     2 * (x - min) / ((max - min) * (mode - min))
	/// } else if mode < x <= max {
	///     2 * (max - x) / ((max - min) * (max - mode))
	/// } else {
	///     0
	/// } )
	/// ```
	fn ln_pdf(&self, x: f64) -> f64 {
		self.pdf(x).ln()
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
fn sample_unchecked<R: rand::Rng + ?Sized>(
	rng: &mut R,
	min: f64,
	max: f64,
	mode: f64,
) -> f64 {
	let f: f64 = rng.random();
	if f < (mode - min) / (max - min) {
		min + (f * (max - min) * (mode - min)).sqrt()
	} else {
		max - ((1.0 - f) * (max - min) * (max - mode)).sqrt()
	}
}
