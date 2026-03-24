use rand::RngExt;
use thiserror::Error;

use core::f64::consts;

use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};
use math::{
	Probability,
	consts::LN_SQRT_2PI,
	function::erf::{erf, erfc, erfc_inv},
};

/// [Levy distribution](https://en.wikipedia.org/wiki/L%C3%A9vy_distribution)
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Levy {
	mu: f64,
	c: f64,
}

/// Errors that can occur when creating [`Levy`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
pub enum LevyError {
	#[error("Location is NaN or infinite")]
	LocationInvalid,
	#[error("Scale is NaN, infinite or nonpositive")]
	ScaleInvalid,
}

impl Levy {
	/// Constructs a new Levy distribution with a location (μ) and dispersion (c)
	///
	/// # Errors
	///
	/// Returns an error:
	///
	/// - If `mu` is NaN or infinite
	/// - If `c` is NaN, infinite or nonpositive
	pub fn new(mu: f64, c: f64) -> Result<Levy, LevyError> {
		if mu.is_nan() || mu.is_infinite() {
			return Err(LevyError::LocationInvalid);
		}
		if c.is_nan() || c.is_infinite() || c <= 0.0 {
			return Err(LevyError::ScaleInvalid);
		}
		Ok(Levy { mu, c })
	}

	/// Location (μ) of the Levy distribution
	pub fn mu(&self) -> f64 {
		self.mu
	}

	/// Dispersion (c) of the Levy distribution
	pub fn c(&self) -> f64 {
		self.c
	}
}

impl core::fmt::Display for Levy {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		write!(f, "Levy({}, {})", self.mu, self.c)
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
	/// `erfc(sqrt(c / (2 ⋅ (x - μ))))` if `x ≤ μ`, else 0
	fn cdf(&self, x: f64) -> f64 {
		if x <= self.mu {
			0.0
		} else if x > 0.0 && x.is_infinite() {
			1.0
		} else {
			erfc(((0.5 * self.c) / (x - self.mu)).sqrt())
		}
	}

	/// `erf(sqrt(c / (2 ⋅ (x - μ))))` if `x > μ` else 1
	fn sf(&self, x: f64) -> f64 {
		if x <= self.mu {
			1.0
		} else if x > 0.0 && x.is_infinite() {
			0.0
		} else {
			erf(((0.5 * self.c) / (x - self.mu)).sqrt())
		}
	}

	/// `μ + c ⋅ (erfc_inv(x)^2) / 2`
	fn inverse_cdf(&self, p: Probability<f64>) -> f64 {
		self.mu + 0.5 * self.c / (erfc_inv(*p).powi(2))
	}

	fn lower(&self) -> f64 {
		self.mu
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Levy {
	/// The mean, always infinity
	fn mean(&self) -> Option<f64> {
		Some(f64::INFINITY)
	}

	/// `μ + c / (2 * erfc_inv(0.5)^2)`
	fn median(&self) -> Option<f64> {
		Some(self.mu + self.c / (2.0 * erfc_inv(0.5).powi(2)))
	}

	/// The variance, always an infinity
	fn variance(&self) -> Option<f64> {
		Some(f64::INFINITY)
	}

	/// The standard deviation, always an infinity
	fn std_dev(&self) -> Option<f64> {
		Some(f64::INFINITY)
	}

	/// `(1 + 3γ + ln(16πc^2))/2`
	fn entropy(&self) -> Option<f64> {
		// CONSTANT_PART = 1.5 * EULER_MASCHERONI + 0.5 * (1.0 + LN_PI + 16.0_f64.ln())
		const CONSTANT_PART: f64 = 3.32448280139689;
		Some(CONSTANT_PART + self.c.ln())
	}
}

impl Mode<Option<f64>> for Levy {
	/// `μ + c/3`
	fn mode(&self) -> Option<f64> {
		Some(self.mu + self.c / 3.0)
	}
}

impl Continuous for Levy {
	/// `(sqrt(c / 2 ⋅ π)) ⋅ e^(-1/2 ⋅ (c / (x - μ))) ⋅ (1 / (x - μ)^(3/2))`
	fn pdf(&self, x: f64) -> f64 {
		if x <= self.mu {
			0.0
		} else {
			let diff = x - self.mu;
			(self.c / consts::TAU).sqrt()
				* (-((0.5 * self.c) / diff)).exp()
				/ diff.powf(1.5)
		}
	}

	/// `½ ⋅ (ln(c) - ln(2 * π) - (c / (x - μ))) - 3/2 ⋅ ln(x - μ)`
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
