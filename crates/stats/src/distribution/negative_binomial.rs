use thiserror::Error;

use math::function::{beta, gamma};

use crate::distribution::{Discrete, DiscreteCDF};
use crate::statistics::{Distribution, Mode};

/// [Negative binomial distribution][wiki]
///
/// The negative binomial distribution is a discrete distribution with two
/// parameters, `r` and `p`.  When `r` is an integer, the negative binomial
/// distribution can be interpreted as the distribution of the number of
/// failures in a sequence of Bernoulli trials that continue until `r` successes
/// occur.  `p` is the probability of success in a single Bernoulli trial.
///
/// `NegativeBinomial` accepts non-integer values for `r`.  This is a
/// generalization of the more common case where `r` is an integer.
///
/// # Examples
///
/// ```
/// use stats::distribution::{NegativeBinomial, Discrete};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let r = NegativeBinomial::new(4.0, 0.5).unwrap();
/// assert_eq!(r.mean().unwrap(), 4.0);
/// assert_almost_eq!(r.pmf(0), 0.0625, 1e-8);
/// assert_almost_eq!(r.pmf(3), 0.15625, 1e-8);
/// ```
///
/// [wiki]: http://en.wikipedia.org/wiki/Negative_binomial_distribution
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NegativeBinomial {
	r: f64,
	p: f64,
}

/// Represents the errors that can occur when creating a [`NegativeBinomial`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Error)]
#[non_exhaustive]
pub enum NegativeBinomialError {
	/// `r` is NaN or less than zero.
	#[error("`r` must positive")]
	RInvalid,

	/// `p` is NaN or not in `[0, 1]`.
	#[error("`p` must lie in [0, 1]")]
	PInvalid,
}

impl NegativeBinomial {
	/// Creates a new [`NegativeBinomial`] distribution
	///
	/// `p` must be `>= 0.0`, `r` must be in `[0, 1]`, neither can be `NaN`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::NegativeBinomial;
	///
	/// NegativeBinomial::new(4.0, 0.5).unwrap();
	/// NegativeBinomial::new(-0.5, 5.0).unwrap_err();
	/// ```
	pub fn new(
		r: f64,
		p: f64,
	) -> Result<NegativeBinomial, NegativeBinomialError> {
		if r.is_nan() || r < 0.0 {
			return Err(NegativeBinomialError::RInvalid);
		}

		if p.is_nan() || !(0.0..=1.0).contains(&p) {
			return Err(NegativeBinomialError::PInvalid);
		}

		Ok(NegativeBinomial { r, p })
	}

	/// Probability of success of a single Bernoulli trial
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::NegativeBinomial;
	///
	/// let nb = NegativeBinomial::new(5.0, 0.5).unwrap();
	/// assert_eq!(nb.p(), 0.5);
	/// ```
	pub fn p(&self) -> f64 {
		self.p
	}

	/// Number of successes of all the trials
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::NegativeBinomial;
	///
	/// let nb = NegativeBinomial::new(5.0, 0.5).unwrap();
	/// assert_eq!(nb.r(), 5.0);
	/// ```
	pub fn r(&self) -> f64 {
		self.r
	}
}

impl core::fmt::Display for NegativeBinomial {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "NB({},{})", self.r, self.p)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<u64> for NegativeBinomial {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> u64 {
		use crate::distribution::{gamma, poisson};

		let lambda = gamma::sample_unchecked(
			r,
			self.r,
			self.p / (1.0 - self.p),
		);
		poisson::sample_unchecked(r, lambda).floor() as u64
	}
}

impl DiscreteCDF for NegativeBinomial {
	/// Calculates the cumulative distribution function for the
	/// negative binomial distribution at `x`.
	///
	/// # Formula
	///
	/// ```text
	/// I_(p)(r, x+1)
	/// ```
	///
	/// where `I_(x)(a, b)` is the regularized incomplete beta function.
	fn cdf(&self, x: u64) -> f64 {
		// XXX: panics?
		beta::beta_reg(self.r, x as f64 + 1.0, self.p).unwrap()
	}

	/// Calculates the survival function for the
	/// negative binomial distribution at `x`
	///
	/// Note that due to extending the distribution to the reals
	/// (allowing positive real values for `r`), while still technically
	/// a discrete distribution the CDF behaves more like that of a
	/// continuous distribution rather than a discrete distribution
	/// (i.e. a smooth graph rather than a step-ladder)
	///
	/// # Formula
	///
	/// ```text
	/// I_(1-p)(x+1, r)
	/// ```
	///
	/// where `I_(x)(a, b)` is the regularized incomplete beta function
	fn sf(&self, x: u64) -> f64 {
		// XXX: panics?
		beta::beta_reg(x as f64 + 1.0, self.r, 1.0 - self.p).unwrap()
	}

	fn lower(&self) -> u64 {
		0
	}

	fn upper(&self) -> u64 {
		u64::MAX
	}
}

impl Distribution for NegativeBinomial {
	/// `r * (1-p) / p`
	fn mean(&self) -> Option<f64> {
		Some(self.r * (1.0 - self.p) / self.p)
	}

	/// `r * (1-p) / p^2`
	fn variance(&self) -> Option<f64> {
		Some(self.r * (1.0 - self.p) / (self.p * self.p))
	}

	/// `(2-p) / sqrt(r * (1-p))`
	fn skewness(&self) -> Option<f64> {
		Some((2.0 - self.p) / f64::sqrt(self.r * (1.0 - self.p)))
	}
}

impl Mode<Option<f64>> for NegativeBinomial {
	/// `floor((r - 1) * (1-p / p))` for `r > 1`, else `0`
	fn mode(&self) -> Option<f64> {
		let mode = if self.r > 1.0 {
			f64::floor((self.r - 1.0) * (1.0 - self.p) / self.p)
		} else {
			0.0
		};
		Some(mode)
	}
}

impl Discrete for NegativeBinomial {
	type T = u64;

	/// Calculates the probability mass function for the negative binomial
	/// distribution at `x`.
	///
	/// # Formula
	///
	/// When `r` is an integer, the formula is:
	///
	/// ```text
	/// (x + r - 1 choose x) * (1 - p)^x * p^r
	/// ```
	///
	/// The general formula for real `r` is:
	///
	/// ```text
	/// Γ(r + x)/(Γ(r) * Γ(x + 1)) * (1 - p)^x * p^r
	/// ```
	///
	/// where Γ(x) is the Gamma function.
	fn pmf(&self, x: u64) -> f64 {
		self.ln_pmf(x).exp()
	}

	/// Calculates the log probability mass function for the negative binomial
	/// distribution at `x`.
	///
	/// # Formula
	///
	/// When `r` is an integer, the formula is:
	///
	/// ```text
	/// ln((x + r - 1 choose x) * (1 - p)^x * p^r)
	/// ```
	///
	/// The general formula for real `r` is:
	///
	/// ```text
	/// ln(Γ(r + x)/(Γ(r) * Γ(x + 1)) * (1 - p)^x * p^r)
	/// ```
	///
	/// where Γ(x) is the Gamma function.
	fn ln_pmf(&self, x: u64) -> f64 {
		let k = x as f64;
		gamma::ln_gamma(self.r + k)
			- gamma::ln_gamma(self.r)
			- gamma::ln_gamma(k + 1.0)
			+ (self.r * self.p.ln())
			+ (k * (-self.p).ln_1p())
	}
}
