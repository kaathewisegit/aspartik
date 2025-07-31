use crate::distribution::{Binomial, BinomialError, Discrete, DiscreteCDF};
use crate::statistics::*;

/// Implements the
/// [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)
/// distribution which is a special case of the
/// [Binomial](https://en.wikipedia.org/wiki/Binomial_distribution)
/// distribution where `n = 1` (referenced [Here](./struct.Binomial.html))
///
/// # Examples
///
/// ```
/// use stats::distribution::{Bernoulli, Discrete};
/// use stats::statistics::Distribution;
///
/// let n = Bernoulli::new(0.5).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.5);
/// assert_eq!(n.pmf(0), 0.5);
/// assert_eq!(n.pmf(1), 0.5);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Bernoulli {
	b: Binomial,
}

impl Bernoulli {
	/// Constructs a new bernoulli distribution with
	/// the given `p` probability of success.
	///
	/// # Errors
	///
	/// Returns an error if `p` is `NaN`, less than `0.0`
	/// or greater than `1.0`
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Bernoulli;
	///
	/// let mut result = Bernoulli::new(0.5);
	/// assert!(result.is_ok());
	///
	/// result = Bernoulli::new(-0.5);
	/// assert!(result.is_err());
	/// ```
	pub fn new(p: f64) -> Result<Bernoulli, BinomialError> {
		Binomial::new(p, 1).map(|b| Bernoulli { b })
	}

	/// Returns the probability of success `p` of the
	/// bernoulli distribution.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Bernoulli;
	///
	/// let n = Bernoulli::new(0.5).unwrap();
	/// assert_eq!(n.p(), 0.5);
	/// ```
	pub fn p(&self) -> f64 {
		self.b.p()
	}

	/// Returns the number of trials `n` of the
	/// bernoulli distribution. Will always be `1.0`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Bernoulli;
	///
	/// let n = Bernoulli::new(0.5).unwrap();
	/// assert_eq!(n.n(), 1);
	/// ```
	pub fn n(&self) -> u64 {
		1
	}
}

impl core::fmt::Display for Bernoulli {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Bernoulli({})", self.p())
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<bool> for Bernoulli {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bool {
		rng.random_bool(self.p())
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Bernoulli {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		let out = rng.sample::<bool, _>(self);
		f64::from(out)
	}
}

impl DiscreteCDF for Bernoulli {
	/// Calculates the cumulative distribution
	/// function for the bernoulli distribution at `x`.
	///
	/// # Formula
	///
	/// ```text
	/// if x >= 1 { 1 }
	/// else { 1 - p }
	/// ```
	fn cdf(&self, x: u64) -> f64 {
		if x >= 1 { 1.0 } else { 1.0 - self.b.p() }
	}

	/// Calculates the survival function for the
	/// bernoulli distribution at `x`.
	///
	/// # Formula
	///
	/// ```text
	/// if x < 0 { 1 }
	/// else if x >= 1 { 0 }
	/// else { p }
	/// ```
	fn sf(&self, x: u64) -> f64 {
		self.b.sf(x)
	}

	fn lower(&self) -> u64 {
		0
	}

	fn upper(&self) -> u64 {
		1
	}
}

impl Distribution for Bernoulli {
	/// Returns the mean of the bernoulli
	/// distribution
	///
	/// # Formula
	///
	/// ```text
	/// p
	/// ```
	fn mean(&self) -> Option<f64> {
		self.b.mean()
	}

	/// Returns the median of the bernoulli
	/// distribution
	///
	/// # Formula
	///
	/// ```text
	/// if p < 0.5 { 0 }
	/// else if p > 0.5 { 1 }
	/// else { 0.5 }
	/// ```
	fn median(&self) -> Option<f64> {
		self.b.median()
	}

	/// Returns the variance of the bernoulli
	/// distribution
	///
	/// # Formula
	///
	/// ```text
	/// p * (1 - p)
	/// ```
	fn variance(&self) -> Option<f64> {
		self.b.variance()
	}

	/// Returns the entropy of the bernoulli
	/// distribution
	///
	/// # Formula
	///
	/// ```text
	/// q = (1 - p)
	/// -q * ln(q) - p * ln(p)
	/// ```
	fn entropy(&self) -> Option<f64> {
		self.b.entropy()
	}

	/// Returns the skewness of the bernoulli
	/// distribution
	///
	/// # Formula
	///
	/// ```text
	/// q = (1 - p)
	/// (1 - 2p) / sqrt(p * q)
	/// ```
	fn skewness(&self) -> Option<f64> {
		self.b.skewness()
	}
}

impl Mode<Option<u64>> for Bernoulli {
	/// Returns the mode of the bernoulli distribution
	///
	/// # Formula
	///
	/// ```text
	/// if p < 0.5 { 0 }
	/// else { 1 }
	/// ```
	fn mode(&self) -> Option<u64> {
		self.b.mode()
	}
}

impl Discrete for Bernoulli {
	type T = u64;

	/// Calculates the probability mass function for the
	/// bernoulli distribution at `x`.
	///
	/// # Formula
	///
	/// ```text
	/// if x == 0 { 1 - p }
	/// else { p }
	/// ```
	fn pmf(&self, x: u64) -> f64 {
		self.b.pmf(x)
	}

	/// Calculates the log probability mass function for the
	/// bernoulli distribution at `x`.
	///
	/// # Formula
	///
	/// ```text
	/// else if x == 0 { ln(1 - p) }
	/// else { ln(p) }
	/// ```
	fn ln_pmf(&self, x: u64) -> f64 {
		self.b.ln_pmf(x)
	}
}
