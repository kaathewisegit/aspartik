use core::f64;
use math::{
	consts::LN_SQRT_2PI,
	function::erf::{erf, erfc, erfc_inv},
};

use crate::distribution::{Continuous, ContinuousCDF};

use crate::statistics::{Distribution, Mode};

/// Implements the [Levy](https://en.wikipedia.org/wiki/L%C3%A9vy_distribution) distribution.
///
/// # Example
///
/// ```
/// use stats::distribution::{Levy, Continuous};
/// use stats::statistics::Distribution;
///
/// let n = Levy::new(1.0, 1.0).unwrap();
/// assert_eq!(n.pdf(0.0), 0.0);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Levy {
	mu: f64,
	c: f64,
}

/// Represents the errors that can occur when creating a [`Levy`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum LevyError {
	/// Location is NaN or infinite
	LocationInvalid,
	/// Scale is NaN, infinite or nonpositive
	ScaleInvalid,
}

impl core::fmt::Display for LevyError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			LevyError::LocationInvalid => {
				write!(f, "location is NaN or infinite")
			}
			LevyError::ScaleInvalid => write!(
				f,
				"scale is NaN, infinite or nonpositive"
			),
		}
	}
}

impl std::error::Error for LevyError {}

impl Levy {
	/// Constructs a new Levy distribution with a location (μ) and dispersion (c)
	///
	/// # Errors
	///
	/// Returns and error if `mu` is NaN or infinite or if `c` is NaN, infinite or nonpositive
	///
	/// # Example
	///
	/// ```
	/// use stats::distribution::Levy;
	///
	/// let mut result = Levy::new(0.0, 1.0);
	/// assert!(result.is_ok());
	///
	/// result = Levy::new(0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(mu: f64, c: f64) -> Result<Levy, LevyError> {
		if mu.is_nan() || mu.is_infinite() {
			return Err(LevyError::LocationInvalid);
		}
		if c.is_nan() || c.is_infinite() || c <= 0.0 {
			return Err(LevyError::ScaleInvalid);
		}
		Ok(Levy { mu, c })
	}

	/// Returns the location (μ) of the Levy distribution
	///
	/// # Example
	///
	/// ```
	/// use stats::distribution::Levy;
	///
	/// let n = Levy::new(1.0, 1.0).unwrap();
	/// assert_eq!(n.mu(), 1.0);
	/// ```
	pub fn mu(&self) -> f64 {
		self.mu
	}

	/// Returns the dispersion (c) of the Levy distribution
	///
	/// # Example
	///
	/// ```
	/// use stats::distribution::Levy;
	///
	/// let n = Levy::new(1.0, 1.0).unwrap();
	/// assert_eq!(n.c(), 1.0);
	/// ```
	pub fn c(&self) -> f64 {
		self.c
	}
}

impl core::fmt::Display for Levy {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		write!(f, "Levy(mu = {}, c = {})", self.mu, self.c)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Levy {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		use rand::distr::OpenClosed01;

		// Inverse transform sampling
		let u: f64 = rng.sample(OpenClosed01);
		self.mu + (0.5 * self.c) / erfc_inv(u).powf(2.0)
	}
}

impl ContinuousCDF for Levy {
	/// Calculates the cumulative distribution function for the Levy distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 0 if x <= μ
	/// erfc(sqrt(c / (2 * (x - μ)))) if x > μ
	/// ```
	///
	/// where `μ` is the location, `c` is the dispersion, and `erfc` is the
	/// complementary error function.
	fn cdf(&self, x: f64) -> f64 {
		if x <= self.mu {
			0.0
		} else if x > 0.0 && x.is_infinite() {
			1.0
		} else {
			erfc(((0.5 * self.c) / (x - self.mu)).sqrt())
		}
	}

	/// Calculates the survival function for the Levy distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 if x <= μ
	/// erf(sqrt(c / (2 * (x - μ)))) if x > μ
	/// ```
	///
	/// where `μ` is the location, `c` is the dispersion, and `erf` is the error
	/// function.
	fn sf(&self, x: f64) -> f64 {
		if x <= self.mu {
			1.0
		} else if x > 0.0 && x.is_infinite() {
			0.0
		} else {
			erf(((0.5 * self.c) / (x - self.mu)).sqrt())
		}
	}

	/// Calculates the inverse cumulative distribution function for the
	/// normal distribution at `x`.
	///
	/// # Panics
	///
	/// If `x < 0.0` or `x > 1.0`
	///
	/// # Formula
	///
	/// ```text
	/// μ + c * (erfc_inv(x)^2)/2
	/// ```
	///
	/// where `μ` is the mean, `σ` is the standard deviation and `erfc_inv` is
	/// the inverse of the complementary error function
	fn inverse_cdf(&self, x: f64) -> f64 {
		if !(0.0..=1.0).contains(&x) {
			panic!("x must be in [0, 1]");
		} else {
			self.mu + 0.5 * self.c / (erfc_inv(x).powf(2.0))
		}
	}

	fn lower(&self) -> f64 {
		self.mu
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Levy {
	/// Returns the mean of the Levy distribution
	///
	/// # Formula
	///
	/// ```text
	/// f64::INFINITY
	/// ```
	fn mean(&self) -> Option<f64> {
		Some(f64::INFINITY)
	}

	/// `μ + c / (2 * erfc_inv(0.5)^2)`, where `μ` is the mean, `c` is the
	/// dispersion and `erfc_inv` is the inverse of the complementary error
	/// function.
	fn median(&self) -> Option<f64> {
		Some(self.mu + self.c / (2.0 * erfc_inv(0.5).powi(2)))
	}

	/// Returns the variance of the Levy distribution
	///
	/// # Formula
	///
	/// ```text
	/// f64::INFINITY
	/// ```
	fn variance(&self) -> Option<f64> {
		Some(f64::INFINITY)
	}

	/// Returns the standard deviation of the Levy distribution
	///
	/// # Formula
	///
	/// ```text
	/// f64::INFINITY
	/// ```
	fn std_dev(&self) -> Option<f64> {
		Some(f64::INFINITY)
	}

	/// Returns the entropy of the Levy distribution
	///
	/// # Formula
	///
	/// ```text
	/// (1 + 3γ + ln(16πc^2))/2
	/// ```
	fn entropy(&self) -> Option<f64> {
		/// CONSTANT_PART = 1.5 * EULER_MASCHERONI + 0.5 * (1.0 + LN_PI + 16.0_f64.ln())
		const CONSTANT_PART: f64 = 3.32448280139689;
		Some(CONSTANT_PART + self.c.ln())
	}
}

impl Mode<Option<f64>> for Levy {
	/// Returns the mode of the Levy distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ + c/3
	/// ```
	///
	/// where `μ` is the mean and `c` is the dispersion.
	fn mode(&self) -> Option<f64> {
		Some(self.mu + self.c / 3.0)
	}
}

impl Continuous for Levy {
	type T = f64;

	/// Calculates the probability density function for the Levy distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (sqrt(c / 2 * π)) * e^(-1/2 * (c / (x - μ))) * (1 / (x - μ)^(3/2))
	/// ```
	///
	/// where `μ` is the mean and `c` is the dispersion.
	fn pdf(&self, x: f64) -> f64 {
		if x <= self.mu {
			0.0
		} else {
			let diff = x - self.mu;
			(self.c / f64::consts::TAU).sqrt()
				* (-((0.5 * self.c) / diff)).exp()
				/ diff.powf(1.5)
		}
	}

	/// Calculates the log probability density function for the Levy distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1/2 * (ln(c) - ln(2 * π) - (c / (x - μ))) - 3/2 * ln(x - μ)
	/// ```
	///
	/// where `μ` is the mean and `σ` is the standard deviation
	fn ln_pdf(&self, x: f64) -> f64 {
		if x <= self.mu {
			f64::NEG_INFINITY
		} else {
			let diff = x - self.mu;
			0.5 * (self.c.ln() - self.c / diff)
				- (1.5 * diff.ln() + LN_SQRT_2PI)
		}
	}
}
