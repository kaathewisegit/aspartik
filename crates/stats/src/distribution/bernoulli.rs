use rand::RngExt;

use std::fmt;

use crate::{
	distribution::{Binomial, Discrete, DiscreteCDF},
	statistics::{Distribution, Mode},
};

/// Implements the [Bernoulli]w[w] distribution
///
/// Bernoulli distribution is a special case of the [Binomial][super::Binomial]
/// distribution where `n` is 1.
///
/// [w]: https://en.wikipedia.org/wiki/Bernoulli_distribution
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Bernoulli {
	b: Binomial,
}

impl Bernoulli {
	/// A new bernoulli distribution with the given probability of success
	pub fn new(p: f64) -> Bernoulli {
		Bernoulli {
			b: Binomial::new(p, 1),
		}
	}

	pub fn p(&self) -> f64 {
		self.b.p()
	}

	/// The number of trials `n` of the bernoulli distribution, always 1
	pub fn n(&self) -> u64 {
		1
	}
}

impl fmt::Display for Bernoulli {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
	/// `1` if `x >= 1` else `1 - p`
	fn cdf(&self, x: u64) -> f64 {
		if x >= 1 { 1.0 } else { 1.0 - self.b.p() }
	}

	/// `1` if `x < 0`, `p` if `x in [0, 1)`, `0` otherwise
	fn sf(&self, x: u64) -> f64 {
		self.b.sf(x)
	}

	/// Always `0`
	fn lower(&self) -> u64 {
		0
	}

	/// Always `1`
	fn upper(&self) -> u64 {
		1
	}
}

impl Distribution for Bernoulli {
	/// `p`
	fn mean(&self) -> Option<f64> {
		self.b.mean()
	}

	/// `0` if `p < 0.5`, `0.5` is `p` is `0.5`, `1` otherwise
	fn median(&self) -> Option<f64> {
		self.b.median()
	}

	/// `p · (1 - p)`
	fn variance(&self) -> Option<f64> {
		self.b.variance()
	}

	/// `-(1 - p) · ln(q) - p · ln(p)`
	fn entropy(&self) -> Option<f64> {
		self.b.entropy()
	}

	/// `(1 - 2p) / sqrt(p · (1 - p))`
	fn skewness(&self) -> Option<f64> {
		self.b.skewness()
	}
}

impl Mode<Option<u64>> for Bernoulli {
	/// `0` if `p < 0.5`, otherwise `1`
	fn mode(&self) -> Option<u64> {
		self.b.mode()
	}
}

impl Discrete for Bernoulli {
	type T = u64;

	/// `1 - p` if `x` is `0`, otherwise `p`
	fn pmf(&self, x: u64) -> f64 {
		self.b.pmf(x)
	}

	/// `ln(1 - p)` if `x` is `0`, otherwise `ln(p)`
	fn ln_pmf(&self, x: u64) -> f64 {
		self.b.ln_pmf(x)
	}
}
