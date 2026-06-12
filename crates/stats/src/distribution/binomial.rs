use computare_special::beta::regularized_incomplete_beta;
use rand::RngExt;

use super::ln_binomial;
use crate::{
	distribution::{Discrete, DiscreteCDF},
	statistics::{Distribution, Mode},
};

/// The [Binomial][w] distribution
///
/// [w]: https://en.wikipedia.org/wiki/Binomial_distribution
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Binomial {
	p: f64,
	n: u64,
}

impl Binomial {
	/// A new binomial distribution with a given `p` probability of success
	/// of `n` trials
	pub fn new(p: f64, n: u64) -> Binomial {
		Binomial { p, n }
	}

	pub fn p(&self) -> f64 {
		self.p
	}

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
	/// `I_(1 - p)(n - x, 1 + x)`
	///
	/// where `I_(x)(a, b)` is the regularized incomplete beta function.
	fn cdf(&self, x: u64) -> f64 {
		if x >= self.n {
			1.0
		} else {
			let k = x;
			regularized_incomplete_beta(
				(self.n - k) as f64,
				k as f64 + 1.0,
				1.0 - self.p,
			)
		}
	}

	/// `I_(p)(x + 1, n - x)`
	///
	/// where `I_(x)(a, b)` is the regularized incomplete beta function
	fn sf(&self, x: u64) -> f64 {
		if x >= self.n {
			0.0
		} else {
			let k = x;
			regularized_incomplete_beta(
				k as f64 + 1.0,
				(self.n - k) as f64,
				self.p,
			)
		}
	}

	/// Always `0`
	fn lower(&self) -> u64 {
		0
	}

	/// Equals `self.n`
	fn upper(&self) -> u64 {
		self.n
	}
}

impl Distribution for Binomial {
	/// `p · n`
	fn mean(&self) -> Option<f64> {
		Some(self.p * self.n as f64)
	}

	/// Floor of `n · p`
	fn median(&self) -> Option<f64> {
		Some((self.p * self.n as f64).floor())
	}

	/// `n · p · (1 - p)`
	fn variance(&self) -> Option<f64> {
		Some(self.p * (1.0 - self.p) * self.n as f64)
	}

	/// `(1 / 2) · ln (2 · π · e · n · p · (1 - p))`
	fn entropy(&self) -> Option<f64> {
		let entr = if self.p == 0.0 || self.p == 1.0 {
			0.0
		} else {
			(0..self.n + 1).fold(0.0, |acc, x| {
				let p = self.pmf(x);
				acc - p * p.ln()
			})
		};
		Some(entr)
	}

	/// `(1 - 2p) / sqrt(n · p · (1 - p)))`
	fn skewness(&self) -> Option<f64> {
		let p = self.p;
		Some((1.0 - 2.0 * p) / (self.n as f64 * p * (1.0 - p)).sqrt())
	}
}

impl Mode<Option<u64>> for Binomial {
	/// Floor of `(n + 1) · p`
	fn mode(&self) -> Option<u64> {
		let mode = if self.p == 0.0 {
			0
		} else if self.p == 1.0 {
			self.n
		} else {
			((self.n as f64 + 1.0) * self.p).floor() as u64
		};
		Some(mode)
	}
}

impl Discrete for Binomial {
	type T = u64;

	/// `C(n, k) · p^k · (1 - p)^(n - k)`
	fn pmf(&self, x: u64) -> f64 {
		if x > self.n {
			0.0
		} else if self.p == 0.0 {
			if x == 0 { 1.0 } else { 0.0 }
		} else if self.p == 1.0 {
			if x == self.n { 1.0 } else { 0.0 }
		} else {
			(ln_binomial(self.n, x)
				+ x as f64 * self.p.ln() + (self.n - x) as f64
				* (1.0 - self.p).ln())
			.exp()
		}
	}

	/// `ln(C(n, k) · p^k · (1 - p)^(n - k))`
	fn ln_pmf(&self, x: u64) -> f64 {
		if x > self.n {
			f64::NEG_INFINITY
		} else if self.p == 0.0 {
			if x == 0 { 0.0 } else { f64::NEG_INFINITY }
		} else if self.p == 1.0 {
			if x == self.n { 0.0 } else { f64::NEG_INFINITY }
		} else {
			ln_binomial(self.n, x)
				+ x as f64 * self.p.ln() + (self.n - x) as f64
				* (1.0 - self.p).ln()
		}
	}
}
