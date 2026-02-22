use rand::RngExt;
use thiserror::Error;

use core::f64;

use crate::{
	distribution::{Discrete, DiscreteCDF},
	statistics::{Distribution, Mode},
};
use math::{Probability, ulps_eq};

/// Implements the
/// [Geometric](https://en.wikipedia.org/wiki/Geometric_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Geometric, Discrete};
/// use stats::statistics::Distribution;
///
/// let n = Geometric::new(0.3).unwrap();
/// assert_eq!(n.mean().unwrap(), 1.0 / 0.3);
/// assert_eq!(n.pmf(1), 0.3);
/// assert_eq!(n.pmf(2), 0.21);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Geometric {
	p: f64,
}

/// Represents the errors that can occur when creating a [`Geometric`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
pub enum GeometricError {
	#[error("The probability is NaN or not in `(0, 1]`")]
	ProbabilityInvalid,
}

impl Geometric {
	/// Constructs a new shifted geometric distribution with a probability
	/// of `p`
	///
	/// # Errors
	///
	/// Returns an error if `p` is not in `(0, 1]`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Geometric;
	///
	/// let mut result = Geometric::new(0.5);
	/// assert!(result.is_ok());
	///
	/// result = Geometric::new(0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(p: f64) -> Result<Geometric, GeometricError> {
		if p <= 0.0 || p > 1.0 || p.is_nan() {
			Err(GeometricError::ProbabilityInvalid)
		} else {
			Ok(Geometric { p })
		}
	}

	/// Returns the probability `p` of the geometric
	/// distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Geometric;
	///
	/// let n = Geometric::new(0.5).unwrap();
	/// assert_eq!(n.p(), 0.5);
	/// ```
	pub fn p(&self) -> f64 {
		self.p
	}
}

impl core::fmt::Display for Geometric {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Geom({})", self.p)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<u64> for Geometric {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> u64 {
		if ulps_eq!(self.p, 1.0) {
			1
		} else {
			let x: f64 = r.sample(rand::distr::OpenClosed01);
			// This cast is safe, because the largest finite value this expression can take is when
			// `x = 1.4e-45` and `1.0 - self.p = 0.9999999999999999`, in which case we get
			// `930262250532780300`, which when casted to a `u64` is `930262250532780288`.
			x.log(1.0 - self.p).ceil() as u64
		}
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Geometric {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
		r.sample::<u64, _>(self) as f64
	}
}

impl DiscreteCDF for Geometric {
	/// Calculates the cumulative distribution function for the geometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 - (1 - p) ^ x
	/// ```
	fn cdf(&self, x: u64) -> f64 {
		if x == 0 {
			0.0
		} else {
			// 1 - (1 - p) ^ x = 1 - exp(log(1 - p)*x)
			//                 = -expm1(log1p(-p)*x))
			//                 = -((-p).ln_1p() * x).exp_m1()
			-((-self.p).ln_1p() * (x as f64)).exp_m1()
		}
	}

	fn inverse_cdf(&self, p: Probability<f64>) -> u64 {
		let p = *p;

		if p == 1.0 || p == 0.0 {
			return 1;
		}

		let approx = (-p).ln_1p() / (-self.p).ln_1p();

		approx.round() as u64
	}

	/// Calculates the survival function for the geometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 - p) ^ x
	/// ```
	fn sf(&self, x: u64) -> f64 {
		// (1-p) ^ x = exp(log(1-p)*x)
		//           = exp(log1p(-p) * x)
		if x == 0 {
			1.0
		} else {
			((-self.p).ln_1p() * (x as f64)).exp()
		}
	}

	fn lower(&self) -> u64 {
		1
	}

	fn upper(&self) -> u64 {
		u64::MAX
	}
}

impl Distribution for Geometric {
	/// Returns the mean of the geometric distribution
	///
	/// # Formula
	///
	/// ```text
	/// 1 / p
	/// ```
	fn mean(&self) -> Option<f64> {
		Some(1.0 / self.p)
	}

	/// Returns the median of the geometric distribution
	///
	/// # Remarks
	///
	/// # Formula
	///
	/// ```text
	/// ceil(-1 / log_2(1 - p))
	/// ```
	fn median(&self) -> Option<f64> {
		Some((-f64::consts::LN_2 / (1.0 - self.p).ln()).ceil())
	}

	/// Returns the standard deviation of the geometric distribution
	///
	/// # Formula
	///
	/// ```text
	/// (1 - p) / p^2
	/// ```
	fn variance(&self) -> Option<f64> {
		Some((1.0 - self.p) / (self.p * self.p))
	}

	/// Returns the entropy of the geometric distribution
	///
	/// # Formula
	///
	/// ```text
	/// (-(1 - p) * log_2(1 - p) - p * log_2(p)) / p
	/// ```
	fn entropy(&self) -> Option<f64> {
		let inv = 1.0 / self.p;
		Some(-inv * (1.0 - self.p).log(2.0) + (inv - 1.0).log(2.0))
	}

	/// Returns the skewness of the geometric distribution
	///
	/// # Formula
	///
	/// ```text
	/// (2 - p) / sqrt(1 - p)
	/// ```
	fn skewness(&self) -> Option<f64> {
		if ulps_eq!(self.p, 1.0) {
			return Some(f64::INFINITY);
		};
		Some((2.0 - self.p) / (1.0 - self.p).sqrt())
	}
}

impl Mode<Option<u64>> for Geometric {
	/// Returns the mode of the geometric distribution
	///
	/// # Formula
	///
	/// ```text
	/// 1
	/// ```
	fn mode(&self) -> Option<u64> {
		Some(1)
	}
}

impl Discrete for Geometric {
	type T = u64;

	/// Calculates the probability mass function for the geometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (1 - p)^(x - 1) * p
	/// ```
	fn pmf(&self, x: u64) -> f64 {
		if x == 0 {
			0.0
		} else {
			(1.0 - self.p).powi(x as i32 - 1) * self.p
		}
	}

	/// Calculates the log probability mass function for the geometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((1 - p)^(x - 1) * p)
	/// ```
	fn ln_pmf(&self, x: u64) -> f64 {
		if x == 0 {
			f64::NEG_INFINITY
		} else if ulps_eq!(self.p, 1.0) && x == 1 {
			0.0
		} else if ulps_eq!(self.p, 1.0) {
			f64::NEG_INFINITY
		} else {
			((x - 1) as f64 * (1.0 - self.p).ln()) + self.p.ln()
		}
	}
}
