use core::cmp;
use math::function::factorial;

use crate::{
	distribution::{Discrete, DiscreteCDF},
	statistics::{Distribution, Mode},
};

/// Implements the
/// [Hypergeometric](http://en.wikipedia.org/wiki/Hypergeometric_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Hypergeometric, Discrete};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let n = Hypergeometric::new(500, 50, 100).unwrap();
/// assert_eq!(n.mean().unwrap(), 10.0);
/// assert_almost_eq!(n.pmf(10), 0.14736784, epsilon = 1e-8);
/// assert_almost_eq!(n.pmf(25), 3.537e-7, epsilon = 1e-10);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Hypergeometric {
	population: u64,
	successes: u64,
	draws: u64,
}

/// Represents the errors that can occur when creating a [`Hypergeometric`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum HypergeometricError {
	/// The number of successes is greater than the population.
	TooManySuccesses,

	/// The number of draws is greater than the population.
	TooManyDraws,
}

impl core::fmt::Display for HypergeometricError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			HypergeometricError::TooManySuccesses => {
				write!(f, "successes > population")
			}
			HypergeometricError::TooManyDraws => {
				write!(f, "draws > population")
			}
		}
	}
}

impl std::error::Error for HypergeometricError {}

impl Hypergeometric {
	/// Constructs a new hypergeometric distribution
	/// with a population (N) of `population`, number
	/// of successes (K) of `successes`, and number of draws
	/// (n) of `draws`.
	///
	/// # Errors
	///
	/// If `successes > population` or `draws > population`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Hypergeometric;
	///
	/// let mut result = Hypergeometric::new(2, 2, 2);
	/// assert!(result.is_ok());
	///
	/// result = Hypergeometric::new(2, 3, 2);
	/// assert!(result.is_err());
	/// ```
	pub fn new(
		population: u64,
		successes: u64,
		draws: u64,
	) -> Result<Hypergeometric, HypergeometricError> {
		if successes > population {
			return Err(HypergeometricError::TooManySuccesses);
		}

		if draws > population {
			return Err(HypergeometricError::TooManyDraws);
		}

		Ok(Hypergeometric {
			population,
			successes,
			draws,
		})
	}

	/// Returns the population size of the hypergeometric
	/// distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Hypergeometric;
	///
	/// let n = Hypergeometric::new(10, 5, 3).unwrap();
	/// assert_eq!(n.population(), 10);
	/// ```
	pub fn population(&self) -> u64 {
		self.population
	}

	/// Returns the number of observed successes of the hypergeometric
	/// distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Hypergeometric;
	///
	/// let n = Hypergeometric::new(10, 5, 3).unwrap();
	/// assert_eq!(n.successes(), 5);
	/// ```
	pub fn successes(&self) -> u64 {
		self.successes
	}

	/// Returns the number of draws of the hypergeometric
	/// distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Hypergeometric;
	///
	/// let n = Hypergeometric::new(10, 5, 3).unwrap();
	/// assert_eq!(n.draws(), 3);
	/// ```
	pub fn draws(&self) -> u64 {
		self.draws
	}

	/// Returns population, successes, and draws in that order
	/// as a tuple of doubles
	fn values_f64(&self) -> (f64, f64, f64) {
		(
			self.population as f64,
			self.successes as f64,
			self.draws as f64,
		)
	}
}

impl core::fmt::Display for Hypergeometric {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(
			f,
			"Hypergeometric({},{},{})",
			self.population, self.successes, self.draws
		)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<u64> for Hypergeometric {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u64 {
		let mut population = self.population as f64;
		let mut successes = self.successes as f64;
		let mut draws = self.draws;
		let mut x = 0;
		loop {
			let p = successes / population;
			let next: f64 = rng.random();
			if next < p {
				x += 1;
				successes -= 1.0;
			}
			population -= 1.0;
			draws -= 1;
			if draws == 0 {
				break;
			}
		}
		x
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Hypergeometric {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		rng.sample::<u64, _>(self) as f64
	}
}

impl DiscreteCDF for Hypergeometric {
	/// Calculates the cumulative distribution function for the hypergeometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 - ((n choose x+1) * (N-n choose K-x-1)) / (N choose K) * 3_F_2(1,
	/// x+1-K, x+1-n; k+2, N+x+2-K-n; 1)
	/// ```
	///
	/// where `N` is population, `K` is successes, `n` is draws,
	/// and `p_F_q` is the
	/// [generalized hypergeometric function](https://en.wikipedia.org/wiki/Generalized_hypergeometric_function)
	///
	/// Calculated as a discrete integral over the probability mass
	/// function evaluated from 0..x+1
	fn cdf(&self, x: u64) -> f64 {
		if x < self.lower() {
			0.0
		} else if x >= self.upper() {
			1.0
		} else {
			let k = x;
			let ln_denom = factorial::ln_binomial(
				self.population,
				self.draws,
			);
			(0..k + 1).fold(0.0, |acc, i| {
				acc + (factorial::ln_binomial(
					self.successes,
					i,
				) + factorial::ln_binomial(
					self.population - self.successes,
					self.draws - i,
				) - ln_denom)
					.exp()
			})
		}
	}

	/// Calculates the survival function for the hypergeometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// 1 - ((n choose x+1) * (N-n choose K-x-1)) / (N choose K) * 3_F_2(1,
	/// x+1-K, x+1-n; x+2, N+x+2-K-n; 1)
	/// ```
	///
	/// where `N` is population, `K` is successes, `n` is draws,
	/// and `p_F_q` is the
	/// [generalized hypergeometric function](https://en.wikipedia.org/wiki/Generalized_hypergeometric_function)
	///
	/// Calculated as a discrete integral over the probability mass
	/// function evaluated from (x+1)..max
	fn sf(&self, x: u64) -> f64 {
		if x < self.lower() {
			1.0
		} else if x >= self.upper() {
			0.0
		} else {
			let k = x;
			let ln_denom = factorial::ln_binomial(
				self.population,
				self.draws,
			);
			(k + 1..=self.upper()).fold(0.0, |acc, i| {
				acc + (factorial::ln_binomial(
					self.successes,
					i,
				) + factorial::ln_binomial(
					self.population - self.successes,
					self.draws - i,
				) - ln_denom)
					.exp()
			})
		}
	}

	fn lower(&self) -> u64 {
		(self.draws + self.successes).saturating_sub(self.population)
	}

	fn upper(&self) -> u64 {
		cmp::min(self.successes, self.draws)
	}
}

impl Distribution for Hypergeometric {
	/// Returns the mean of the hypergeometric distribution
	///
	/// # None
	///
	/// If `N` is `0`
	///
	/// # Formula
	///
	/// ```text
	/// K * n / N
	/// ```
	///
	/// where `N` is population, `K` is successes, and `n` is draws
	fn mean(&self) -> Option<f64> {
		if self.population == 0 {
			None
		} else {
			Some(self.successes as f64 * self.draws as f64
				/ self.population as f64)
		}
	}

	/// Returns the variance of the hypergeometric distribution
	///
	/// # None
	///
	/// If `N <= 1`
	///
	/// # Formula
	///
	/// ```text
	/// n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))
	/// ```
	///
	/// where `N` is population, `K` is successes, and `n` is draws
	fn variance(&self) -> Option<f64> {
		if self.population <= 1 {
			None
		} else {
			let (population, successes, draws) = self.values_f64();
			let val = draws
				* successes * (population - draws)
				* (population - successes) / (population
				* population
				* (population - 1.0));
			Some(val)
		}
	}

	/// Returns the skewness of the hypergeometric distribution
	///
	/// # None
	///
	/// If `N <= 2`
	///
	/// # Formula
	///
	/// ```text
	/// ((N - 2K) * (N - 1)^(1 / 2) * (N - 2n)) / ([n * K * (N - K) * (N -
	/// n)]^(1 / 2) * (N - 2))
	/// ```
	///
	/// where `N` is population, `K` is successes, and `n` is draws
	fn skewness(&self) -> Option<f64> {
		if self.population <= 2 {
			None
		} else {
			let (population, successes, draws) = self.values_f64();
			let val = (population - 1.0).sqrt()
				* (population - 2.0 * draws) * (population
				- 2.0 * successes) / ((draws
				* successes
				* (population - successes)
				* (population - draws))
				.sqrt() * (population
				- 2.0));
			Some(val)
		}
	}
}

impl Mode<Option<u64>> for Hypergeometric {
	/// Returns the mode of the hypergeometric distribution
	///
	/// # Formula
	///
	/// ```text
	/// floor((n + 1) * (k + 1) / (N + 2))
	/// ```
	///
	/// where `N` is population, `K` is successes, and `n` is draws
	fn mode(&self) -> Option<u64> {
		Some(((self.draws + 1) * (self.successes + 1))
			/ (self.population + 2))
	}
}

impl Discrete for Hypergeometric {
	type T = u64;

	/// Calculates the probability mass function for the hypergeometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// (K choose x) * (N-K choose n-x) / (N choose n)
	/// ```
	///
	/// where `N` is population, `K` is successes, and `n` is draws
	fn pmf(&self, x: u64) -> f64 {
		if x > self.draws {
			0.0
		} else {
			factorial::binomial(self.successes, x)
				* factorial::binomial(
					self.population - self.successes,
					self.draws - x,
				) / factorial::binomial(self.population, self.draws)
		}
	}

	/// Calculates the log probability mass function for the hypergeometric
	/// distribution at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln((K choose x) * (N-K choose n-x) / (N choose n))
	/// ```
	///
	/// where `N` is population, `K` is successes, and `n` is draws
	fn ln_pmf(&self, x: u64) -> f64 {
		factorial::ln_binomial(self.successes, x)
			+ factorial::ln_binomial(
				self.population - self.successes,
				self.draws - x,
			) - factorial::ln_binomial(self.population, self.draws)
	}
}
