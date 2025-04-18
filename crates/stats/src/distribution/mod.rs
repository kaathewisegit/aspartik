//! Defines common interfaces for interacting with statistical distributions
//! and provides
//! concrete implementations for a variety of distributions.
use num_traits::{Float, Num, NumAssignOps, One};

mod bernoulli;
mod beta;
mod binomial;
#[cfg(feature = "std")]
mod categorical;
mod cauchy;
mod chi;
mod chi_squared;
mod discrete_uniform;
mod erlang;
mod exponential;
mod fisher_snedecor;
mod gamma;
mod geometric;
mod gumbel;
mod hypergeometric;
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
pub use binomial::{Binomial, BinomialError};
#[cfg(feature = "std")]
pub use categorical::{Categorical, CategoricalError};
pub use cauchy::{Cauchy, CauchyError};
pub use chi::{Chi, ChiError};
pub use chi_squared::ChiSquared;
pub use discrete_uniform::{DiscreteUniform, DiscreteUniformError};
pub use erlang::Erlang;
pub use exponential::{Exp, ExpError};
pub use fisher_snedecor::{FisherSnedecor, FisherSnedecorError};
pub use gamma::{Gamma, GammaError};
pub use geometric::{Geometric, GeometricError};
pub use gumbel::{Gumbel, GumbelError};
pub use hypergeometric::{Hypergeometric, HypergeometricError};
pub use inverse_gamma::{InverseGamma, InverseGammaError};
pub use laplace::{Laplace, LaplaceError};
pub use levy::{Levy, LevyError};
pub use log_normal::{LogNormal, LogNormalError};
pub use negative_binomial::{NegativeBinomial, NegativeBinomialError};
pub use normal::{Normal, NormalError};
pub use pareto::{Pareto, ParetoError};
pub use poisson::{Poisson, PoissonError};
pub use students_t::{StudentsT, StudentsTError};
pub use triangular::{Triangular, TriangularError};
pub use uniform::{Uniform, UniformError};
pub use weibull::{Weibull, WeibullError};

/// The `Continuous` trait  provides an interface for interacting with
/// continuous statistical distributions
///
/// # Remarks
///
/// All methods provided by the `Continuous` trait are unchecked, meaning
/// they can panic if in an invalid state or encountering invalid input
/// depending on the implementing distribution.
pub trait Continuous {
	type T;

	/// Returns the probability density function calculated at `x` for a given
	/// distribution.
	/// May panic depending on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Continuous, Uniform};
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(1.0, n.pdf(0.5));
	/// ```
	fn pdf(&self, x: Self::T) -> f64;

	/// Returns the log of the probability density function calculated at `x`
	/// for a given distribution.
	/// May panic depending on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Continuous, Uniform};
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.0, n.ln_pdf(0.5));
	/// ```
	fn ln_pdf(&self, x: Self::T) -> f64;
}

/// The `ContinuousCDF` trait is used to specify an interface for univariate
/// distributions for which cdf float arguments are sensible.
pub trait ContinuousCDF: Continuous
where
	Self::T: Float,
{
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
	fn cdf(&self, x: Self::T) -> f64;

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
	fn sf(&self, x: Self::T) -> f64 {
		1.0 - self.cdf(x)
	}

	/// Due to issues with rounding and floating-point accuracy the default
	/// implementation may be ill-behaved.
	/// Specialized inverse cdfs should be used whenever possible.
	/// Performs a binary search on the domain of `cdf` to obtain an approximation
	/// of `F^-1(p) := inf { x | F(x) >= p }`. Needless to say, performance may
	/// may be lacking.
	#[doc(alias = "quantile function")]
	#[doc(alias = "quantile")]
	fn inverse_cdf(&self, p: f64) -> Self::T {
		if p == 0.0 {
			return self.lower();
		};
		if p == 1.0 {
			return self.upper();
		};
		let two = Self::T::one() + Self::T::one();
		let mut high = two;
		let mut low = -high;
		while self.cdf(low) > p {
			low = low + low;
		}
		while self.cdf(high) < p {
			high = high + high;
		}
		let mut i = 16;
		while i != 0 {
			let mid = (high + low) / two;
			if self.cdf(mid) >= p {
				high = mid;
			} else {
				low = mid;
			}
			i -= 1;
		}
		(high + low) / two
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
	/// May panic depending on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Discrete, Binomial};
	/// use stats::assert_almost_eq;
	///
	/// let n = Binomial::new(0.5, 10).unwrap();
	/// assert_almost_eq!(n.pmf(5), 0.24609375, 1e-15);
	/// ```
	fn pmf(&self, x: Self::T) -> f64;

	/// Returns the log of the probability mass function calculated at `x` for
	/// a given distribution.
	/// May panic depending on the implementor.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Discrete, Binomial};
	/// use stats::assert_almost_eq;
	///
	/// let n = Binomial::new(0.5, 10).unwrap();
	/// assert_almost_eq!(n.ln_pmf(5), (0.24609375f64).ln(), 1e-15);
	/// ```
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

	/// Due to issues with rounding and floating-point accuracy the default implementation may be ill-behaved
	/// Specialized inverse cdfs should be used whenever possible.
	///
	/// # Panics
	/// this default impl panics if provided `p` not on interval [0.0, 1.0]
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

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<'_, PyModule>> {
	let m = PyModule::new(py, "distributions")?;

	m.add_class::<Gamma>()?;
	m.add_class::<Poisson>()?;
	m.add_class::<Uniform>()?;
	m.add_class::<Exp>()?;
	m.add_class::<LogNormal>()?;

	m.add_class::<GammaError>()?;
	m.add_class::<PoissonError>()?;
	m.add_class::<UniformError>()?;
	m.add_class::<ExpError>()?;
	m.add_class::<LogNormalError>()?;

	Ok(m)
}
