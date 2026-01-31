use thiserror::Error;

use crate::{
	distribution::{Discrete, DiscreteCDF},
	statistics::{Distribution, Mode},
};
use math::{
	function::{beta, factorial},
	ulps_eq,
};

/// Implements the
/// [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Binomial, Discrete};
/// use stats::statistics::Distribution;
///
/// let n = Binomial::new(0.5, 5).unwrap();
/// assert_eq!(n.mean().unwrap(), 2.5);
/// assert_eq!(n.pmf(0), 0.03125);
/// assert_eq!(n.pmf(3), 0.3125);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Binomial {
	p: f64,
	n: u64,
}

/// Represents the errors that can occur when creating a [`Binomial`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
pub enum BinomialError {
	#[error("The probability is NaN or not in `[0, 1]`")]
	ProbabilityInvalid,
}

impl Binomial {
	/// Constructs a new binomial distribution
	/// with a given `p` probability of success of `n`
	/// trials.
	///
	/// # Errors
	///
	/// Returns an error if `p` is `NaN`, less than `0.0`,
	/// greater than `1.0`, or if `n` is less than `0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Binomial;
	///
	/// let mut result = Binomial::new(0.5, 5);
	/// assert!(result.is_ok());
	///
	/// result = Binomial::new(-0.5, 5);
	/// assert!(result.is_err());
	/// ```
	pub fn new(p: f64, n: u64) -> Result<Binomial, BinomialError> {
		if p.is_nan() || !(0.0..=1.0).contains(&p) {
			Err(BinomialError::ProbabilityInvalid)
		} else {
			Ok(Binomial { p, n })
		}
	}

	/// Returns the probability of success `p` of
	/// the binomial distribution.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Binomial;
	///
	/// let n = Binomial::new(0.5, 5).unwrap();
	/// assert_eq!(n.p(), 0.5);
	/// ```
	pub fn p(&self) -> f64 {
		self.p
	}

	/// Returns the number of trials `n` of the
	/// binomial distribution.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Binomial;
	///
	/// let n = Binomial::new(0.5, 5).unwrap();
	/// assert_eq!(n.n(), 5);
	/// ```
	pub fn n(&self) -> u64 {
		self.n
	}
}

impl core::fmt::Display for Binomial {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Bin({},{})", self.p, self.n)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<u64> for Binomial {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
		(0..self.n).fold(0, |acc, _| {
			let n: f64 = rng.random();
			if n < self.p { acc + 1 } else { acc }
		})
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Binomial {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		rng.sample::<u64, _>(self) as f64
	}
}

impl DiscreteCDF for Binomial {
	/// Calculates the cumulative distribution function for the
	/// binomial distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// I_(1 - p)(n - x, 1 + x)
	/// ```
	///
	/// where `I_(x)(a, b)` is the regularized incomplete beta function
	fn cdf(&self, x: u64) -> f64 {
		if x >= self.n {
			1.0
		} else {
			let k = x;
			beta::beta_reg(
				(self.n - k) as f64,
				k as f64 + 1.0,
				1.0 - self.p,
			)
			// XXX: panics?
			.unwrap()
		}
	}

	/// Calculates the survival function for the
	/// binomial distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// I_(p)(x + 1, n - x)
	/// ```
	///
	/// where `I_(x)(a, b)` is the regularized incomplete beta function
	fn sf(&self, x: u64) -> f64 {
		if x >= self.n {
			0.0
		} else {
			let k = x;
			beta::beta_reg(
				k as f64 + 1.0,
				(self.n - k) as f64,
				self.p,
			)
			// XXX: panics?
			.unwrap()
		}
	}

	fn lower(&self) -> u64 {
		0
	}

	fn upper(&self) -> u64 {
		self.n
	}
}

impl Distribution for Binomial {
	/// Returns the mean of the binomial distribution
	///
	/// # Formula
	///
	/// ```text
	/// p * n
	/// ```
	fn mean(&self) -> Option<f64> {
		Some(self.p * self.n as f64)
	}

	/// Returns the median of the binomial distribution
	///
	/// # Formula
	///
	/// ```text
	/// floor(n * p)
	/// ```
	fn median(&self) -> Option<f64> {
		Some((self.p * self.n as f64).floor())
	}

	/// Returns the variance of the binomial distribution
	///
	/// # Formula
	///
	/// ```text
	/// n * p * (1 - p)
	/// ```
	fn variance(&self) -> Option<f64> {
		Some(self.p * (1.0 - self.p) * self.n as f64)
	}

	/// Returns the entropy of the binomial distribution
	///
	/// # Formula
	///
	/// ```text
	/// (1 / 2) * ln (2 * π * e * n * p * (1 - p))
	/// ```
	fn entropy(&self) -> Option<f64> {
		let entr = if self.p == 0.0 || ulps_eq!(self.p, 1.0) {
			0.0
		} else {
			(0..self.n + 1).fold(0.0, |acc, x| {
				let p = self.pmf(x);
				acc - p * p.ln()
			})
		};
		Some(entr)
	}

	/// Returns the skewness of the binomial distribution
	///
	/// # Formula
	///
	/// ```text
	/// (1 - 2p) / sqrt(n * p * (1 - p)))
	/// ```
	fn skewness(&self) -> Option<f64> {
		Some((1.0 - 2.0 * self.p)
			/ (self.n as f64 * self.p * (1.0 - self.p)).sqrt())
	}
}

impl Mode<Option<u64>> for Binomial {
	/// Returns the mode for the binomial distribution
	///
	/// # Formula
	///
	/// ```text
	/// floor((n + 1) * p)
	/// ```
	fn mode(&self) -> Option<u64> {
		let mode = if self.p == 0.0 {
			0
		} else if ulps_eq!(self.p, 1.0) {
			self.n
		} else {
			((self.n as f64 + 1.0) * self.p).floor() as u64
		};
		Some(mode)
	}
}

impl Discrete for Binomial {
	type T = u64;

	/// Calculates the probability mass function for the binomial
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (n choose k) * p^k * (1 - p)^(n - k)
	/// ```
	fn pmf(&self, x: u64) -> f64 {
		if x > self.n {
			0.0
		} else if self.p == 0.0 {
			if x == 0 { 1.0 } else { 0.0 }
		} else if ulps_eq!(self.p, 1.0) {
			if x == self.n { 1.0 } else { 0.0 }
		} else {
			(factorial::ln_binomial(self.n, x)
				+ x as f64 * self.p.ln() + (self.n - x) as f64
				* (1.0 - self.p).ln())
			.exp()
		}
	}

	/// Calculates the log probability mass function for the binomial
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((n choose k) * p^k * (1 - p)^(n - k))
	/// ```
	fn ln_pmf(&self, x: u64) -> f64 {
		if x > self.n {
			f64::NEG_INFINITY
		} else if self.p == 0.0 {
			if x == 0 { 0.0 } else { f64::NEG_INFINITY }
		} else if ulps_eq!(self.p, 1.0) {
			if x == self.n { 0.0 } else { f64::NEG_INFINITY }
		} else {
			factorial::ln_binomial(self.n, x)
				+ x as f64 * self.p.ln() + (self.n - x) as f64
				* (1.0 - self.p).ln()
		}
	}
}
