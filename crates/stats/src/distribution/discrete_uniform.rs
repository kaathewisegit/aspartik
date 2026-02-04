use rand::RngExt;
use thiserror::Error;

use crate::{
	distribution::{Discrete, DiscreteCDF},
	statistics::{Distribution, Mode},
};

/// Implements the [Discrete
/// Uniform](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{DiscreteUniform, Discrete};
/// use stats::statistics::Distribution;
///
/// let n = DiscreteUniform::new(0, 5).unwrap();
/// assert_eq!(n.mean().unwrap(), 2.5);
/// assert_eq!(n.pmf(3), 1.0 / 6.0);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct DiscreteUniform {
	min: i64,
	max: i64,
}

/// Represents the errors that can occur when creating a [`DiscreteUniform`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
pub enum DiscreteUniformError {
	#[error("Maximum is less than minimum")]
	MinMaxInvalid,
}

impl DiscreteUniform {
	/// Constructs a new discrete uniform distribution with a minimum value
	/// of `min` and a maximum value of `max`.
	///
	/// # Errors
	///
	/// Returns an error if `max < min`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::DiscreteUniform;
	///
	/// let mut result = DiscreteUniform::new(0, 5);
	/// assert!(result.is_ok());
	///
	/// result = DiscreteUniform::new(5, 0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(
		min: i64,
		max: i64,
	) -> Result<DiscreteUniform, DiscreteUniformError> {
		if max < min {
			Err(DiscreteUniformError::MinMaxInvalid)
		} else {
			Ok(DiscreteUniform { min, max })
		}
	}

	/// Returns the minimum value in the domain of the discrete uniform
	/// distribution
	///
	/// # Remarks
	///
	/// This is the same value as the minimum passed into the constructor
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::DiscreteUniform;
	///
	/// let n = DiscreteUniform::new(0, 5).unwrap();
	/// assert_eq!(n.min(), 0);
	/// ```
	pub fn min(&self) -> i64 {
		self.min
	}

	/// Returns the maximum value in the domain of the discrete uniform
	/// distribution
	///
	/// # Remarks
	///
	/// This is the same value as the maximum passed into the constructor
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::DiscreteUniform;
	///
	/// let n = DiscreteUniform::new(0, 5).unwrap();
	/// assert_eq!(n.max(), 5);
	/// ```
	pub fn max(&self) -> i64 {
		self.max
	}
}

impl core::fmt::Display for DiscreteUniform {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Uni([{}, {}])", self.min, self.max)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<i64> for DiscreteUniform {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> i64 {
		rng.random_range(self.min..=self.max)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for DiscreteUniform {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		rng.sample::<i64, _>(self) as f64
	}
}

impl DiscreteCDF for DiscreteUniform {
	/// Calculates the cumulative distribution function for the
	/// discrete uniform distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (floor(x) - min + 1) / (max - min + 1)
	/// ```
	fn cdf(&self, x: i64) -> f64 {
		if x < self.min {
			0.0
		} else if x >= self.max {
			1.0
		} else {
			let lower = self.min as f64;
			let upper = self.max as f64;
			let ans = (x as f64 - lower + 1.0)
				/ (upper - lower + 1.0);
			if ans > 1.0 { 1.0 } else { ans }
		}
	}

	fn sf(&self, x: i64) -> f64 {
		// 1.0 - self.cdf(x)
		if x < self.min {
			1.0
		} else if x >= self.max {
			0.0
		} else {
			let lower = self.min as f64;
			let upper = self.max as f64;
			let ans = (upper - x as f64) / (upper - lower + 1.0);
			if ans > 1.0 { 1.0 } else { ans }
		}
	}

	fn lower(&self) -> i64 {
		self.min
	}

	fn upper(&self) -> i64 {
		self.max
	}
}

impl Distribution for DiscreteUniform {
	/// Returns the mean of the discrete uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// (min + max) / 2
	/// ```
	fn mean(&self) -> Option<f64> {
		Some((self.min + self.max) as f64 / 2.0)
	}

	/// Returns the median of the discrete uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// (max + min) / 2
	/// ```
	fn median(&self) -> Option<f64> {
		Some((self.min + self.max) as f64 / 2.0)
	}

	/// Returns the variance of the discrete uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// ((max - min + 1)^2 - 1) / 12
	/// ```
	fn variance(&self) -> Option<f64> {
		let diff = (self.max - self.min) as f64;
		Some(((diff + 1.0) * (diff + 1.0) - 1.0) / 12.0)
	}

	/// Returns the entropy of the discrete uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// ln(max - min + 1)
	/// ```
	fn entropy(&self) -> Option<f64> {
		let diff = (self.max - self.min) as f64;
		Some((diff + 1.0).ln())
	}

	/// Returns the skewness of the discrete uniform distribution
	///
	/// # Formula
	///
	/// ```text
	/// 0
	/// ```
	fn skewness(&self) -> Option<f64> {
		Some(0.0)
	}
}

impl Mode<Option<i64>> for DiscreteUniform {
	/// Returns the mode for the discrete uniform distribution
	///
	/// # Remarks
	///
	/// Since every element has an equal probability, mode simply
	/// returns the middle element
	///
	/// # Formula
	///
	/// ```text
	/// N/A // (max + min) / 2 for the middle element
	/// ```
	fn mode(&self) -> Option<i64> {
		Some(((self.min + self.max) as f64 / 2.0).floor() as i64)
	}
}

impl Discrete for DiscreteUniform {
	type T = i64;

	/// Calculates the probability mass function for the discrete uniform
	/// distribution at `x`
	///
	/// # Remarks
	///
	/// Returns `0.0` if `x` is not in `[min, max]`
	///
	/// # Formula
	///
	/// ```text
	/// 1 / (max - min + 1)
	/// ```
	fn pmf(&self, x: i64) -> f64 {
		if x >= self.min && x <= self.max {
			1.0 / (self.max - self.min + 1) as f64
		} else {
			0.0
		}
	}

	/// Calculates the log probability mass function for the discrete uniform
	/// distribution at `x`
	///
	/// # Remarks
	///
	/// Returns `f64::NEG_INFINITY` if `x` is not in `[min, max]`
	///
	/// # Formula
	///
	/// ```text
	/// ln(1 / (max - min + 1))
	/// ```
	fn ln_pmf(&self, x: i64) -> f64 {
		if x >= self.min && x <= self.max {
			-((self.max - self.min + 1) as f64).ln()
		} else {
			f64::NEG_INFINITY
		}
	}
}
