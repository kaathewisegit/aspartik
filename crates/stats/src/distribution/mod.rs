//! Defines common interfaces for interacting with statistical distributions and
//! provides concrete implementations for a variety of distributions.
use computare_special::gamma::ln_gamma;
use num_traits::{Num, NumAssignOps, One};

mod bernoulli;
mod beta;
mod binomial;
mod categorical;
mod cauchy;
mod chi;
mod chi_squared;
mod discrete_uniform;
mod erlang;
mod exponential;
mod gamma;
mod geometric;
mod gumbel;
#[macro_use]
mod internal;
mod inverse_gamma;
mod laplace;
mod levy;
mod log_normal;
mod negative_binomial;
mod normal;
mod pareto;
mod poisson;
mod students_t;
mod triangular;
mod uniform;
mod weibull;
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
mod ziggurat;
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
mod ziggurat_tables;

pub use bernoulli::Bernoulli;
pub use beta::{Beta, BetaError};
pub use binomial::Binomial;
pub use categorical::{Categorical, CategoricalError};
pub use cauchy::{Cauchy, CauchyError};
pub use chi::Chi;
pub use chi_squared::ChiSquared;
pub use discrete_uniform::{DiscreteUniform, DiscreteUniformError};
pub use erlang::Erlang;
pub use exponential::Exp;
pub use gamma::{Gamma, GammaError};
pub use geometric::{Geometric, GeometricError};
pub use gumbel::{Gumbel, GumbelError};
pub use inverse_gamma::{InverseGamma, InverseGammaError};
pub use laplace::{Laplace, LaplaceError};
pub use levy::{Levy, LevyError};
pub use log_normal::{LogNormal, LogNormalError};
pub use negative_binomial::{NegativeBinomial, NegativeBinomialError};
pub use normal::{Normal, NormalError};
pub use pareto::{Pareto, ParetoError};
pub use poisson::Poisson;
pub use students_t::{StudentsT, StudentsTError};
pub use triangular::{Triangular, TriangularError};
pub use uniform::{Uniform, UniformError};
pub use weibull::{Weibull, WeibullError};

/// An interface for interacting with continuous statistical distributions
///
///
/// # Remarks
///
/// All methods provided by the `Continuous` trait are unchecked, meaning they
/// can panic if in an invalid state or encountering invalid input depending on
/// the implementing distribution.
pub trait Continuous {
	/// Returns the probability density function calculated at `x` for a
	/// given distribution.  May panic depending on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Continuous, Uniform};
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(1.0, n.pdf(0.5));
	/// ```
	fn pdf(&self, x: f64) -> f64;

	/// Returns the log of the probability density function calculated at
	/// `x` for a given distribution.  May panic depending on the
	/// implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Continuous, Uniform};
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.0, n.ln_pdf(0.5));
	/// ```
	fn ln_pdf(&self, x: f64) -> f64;
}

/// The `ContinuousCDF` trait is used to specify an interface for univariate
/// distributions for which cdf float arguments are sensible.
pub trait ContinuousCDF: Continuous {
	/// Returns the cumulative distribution function calculated
	/// at `x` for a given distribution. May panic depending
	/// on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{ContinuousCDF, Uniform};
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.5, n.cdf(0.5));
	/// ```
	fn cdf(&self, x: f64) -> f64;

	/// Returns the survival function calculated
	/// at `x` for a given distribution. May panic depending
	/// on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{ContinuousCDF, Uniform};
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.5, n.sf(0.5));
	/// ```
	fn sf(&self, x: f64) -> f64 {
		1.0 - self.cdf(x)
	}

	/// The quantile function
	///
	/// The default implementation may be ill-behaved due to issues with
	/// rounding and floating-point accuracy.  Specialized inverse cdfs
	/// should be used whenever possible.  The default implementation
	/// performs a binary search on the domain of `cdf` to obtain an
	/// approximation of `F^-1(p) := inf { x | F(x) >= p }`. Needless to
	/// say, performance may may be lacking.
	#[doc(alias = "quantile function")]
	#[doc(alias = "quantile")]
	fn inverse_cdf(&self, p: f64) -> f64 {
		if p == 0.0 {
			return self.lower();
		};
		if p == 1.0 {
			return self.upper();
		};
		let mut high = 2.0;
		let mut low = -2.0;
		while self.cdf(low) > p {
			low = low + low;
		}
		while self.cdf(high) < p {
			high = high + high;
		}
		let mut i = 16;
		while i != 0 {
			let mid = (high + low) / 2.0;
			if self.cdf(mid) >= p {
				high = mid;
			} else {
				low = mid;
			}
			i -= 1;
		}
		(high + low) / 2.0
	}

	/// The lower bound on the values returned by the distribution
	///
	/// Represents the start of the support.
	fn lower(&self) -> f64;

	/// The upper bound on the values returned by the distribution
	///
	/// Represents the end of the support.  Rays are represented via the
	/// maximum value of the `T` type (infinity for floats and the maximum
	/// possible value for integers).
	fn upper(&self) -> f64;
}

/// The `Discrete` trait provides an interface for interacting with discrete
/// statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Discrete` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution.
pub trait Discrete {
	type T;

	/// Returns the probability mass function calculated at `x` for a given
	/// distribution.
	///
	/// May panic depending on the implementor.
	fn pmf(&self, x: Self::T) -> f64;

	/// Returns the log of the probability mass function calculated at `x`
	/// for a given distribution.
	///
	/// May panic depending on the implementor.
	fn ln_pmf(&self, x: Self::T) -> f64;
}

/// The `DiscreteCDF` trait is used to specify an interface for univariate
/// discrete distributions.
pub trait DiscreteCDF: Discrete
where
	Self::T: Sized + Num + One + Ord + Clone + NumAssignOps,
{
	/// Returns the cumulative distribution function calculated
	/// at `x` for a given distribution. May panic depending
	/// on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{DiscreteCDF, DiscreteUniform};
	///
	/// let n = DiscreteUniform::new(1, 10).unwrap();
	/// assert_eq!(0.6, n.cdf(6));
	/// ```
	fn cdf(&self, x: Self::T) -> f64;

	/// Returns the survival function calculated at `x` for
	/// a given distribution. May panic depending on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{DiscreteCDF, DiscreteUniform};
	///
	/// let n = DiscreteUniform::new(1, 10).unwrap();
	/// assert_eq!(0.4, n.sf(6));
	/// ```
	fn sf(&self, x: Self::T) -> f64 {
		1.0 - self.cdf(x)
	}

	/// Due to issues with rounding and floating-point accuracy the default
	/// implementation may be ill-behaved Specialized inverse cdfs should be
	/// used whenever possible.
	fn inverse_cdf(&self, p: f64) -> Self::T {
		if p <= self.cdf(self.lower()) {
			return self.lower();
		} else if p == 1.0 {
			return self.upper();
		} else if !(0.0..=1.0).contains(&p) {
			panic!("p must be on [0, 1]")
		}

		let two = Self::T::one() + Self::T::one();
		let mut ub = two.clone();
		let lb = self.lower();
		while self.cdf(ub.clone()) < p {
			ub *= two.clone();
		}

		internal::integral_bisection_search(
			|p| self.cdf(p.clone()),
			p,
			lb,
			ub,
		)
		.unwrap()
	}

	/// The lower bound on the values returned by the distribution
	///
	/// Represents the start of the support.
	fn lower(&self) -> Self::T;

	/// The upper bound on the values returned by the distribution
	///
	/// Represents the end of the support.  Rays are represented via the
	/// maximum value of the `T` type (infinity for floats and the maximum
	/// possible value for integers).
	fn upper(&self) -> Self::T;
}

/// `ln(x!)`
///
/// Returns `0.0` if `x <= 1`.
fn ln_factorial(x: u64) -> f64 {
	pub const MAX_FACTORIAL: usize = 170;
	// Initialization for pre-computed cache of 171 factorial values
	// 0!...170!
	const FCACHE: [f64; MAX_FACTORIAL + 1] = {
		let mut fcache = [1.0; MAX_FACTORIAL + 1];

		// `const` only allow while loops (because `next` on `Iterator` isn't
		// `const`)
		let mut i = 1;
		while i < MAX_FACTORIAL + 1 {
			fcache[i] = fcache[i - 1] * i as f64;
			i += 1;
		}

		fcache
	};

	let x = x as usize;
	FCACHE.get(x)
		.map_or_else(|| ln_gamma(x as f64 + 1.0), |&fac| fac.ln())
}

/// Natural logarithm of the binomial coefficient
///
/// Returns negative infinity if `k > n`.
fn ln_binomial(n: u64, k: u64) -> f64 {
	if k > n {
		f64::NEG_INFINITY
	} else {
		ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
	}
}
