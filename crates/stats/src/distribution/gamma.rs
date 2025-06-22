use approx::ulps_eq;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

#[cfg(feature = "python")]
use crate::python_macros::{impl_pyerr, impl_pymethods};
use crate::{
	distribution::{Continuous, ContinuousCDF},
	function::gamma,
	prec,
	statistics::{Distribution, Mode},
};

/// Implements the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{Gamma, Continuous};
/// use stats::statistics::Distribution;
/// use stats::assert_almost_eq;
///
/// let n = Gamma::new(3.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 3.0);
/// assert_almost_eq!(n.pdf(2.0), 0.270670566473225383788, 1e-15);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub struct Gamma {
	shape: f64,
	rate: f64,
}

#[cfg(feature = "python")]
impl_pymethods! {for Gamma;
	new(shape: f64, rate: f64) throws GammaError;
	get(py_shape) shape: f64;
	get(py_rate) rate: f64;
	repr("Gamma(shape={}, rate={})", shape, rate);
	Continuous;
	ContinuousCDF;
	Distribution;
	sample;
	pickle(shape, rate);
}

/// Represents the errors that can occur when creating a [`Gamma`].
#[derive(Debug, Copy, Clone, PartialEq, Error)]
#[non_exhaustive]
#[cfg_attr(
	feature = "python",
	pyclass(module = "aspartik.stats.distributions", frozen, eq, str)
)]
pub enum GammaError {
	#[error("Shape is NaN, zero or less than zero")]
	ShapeInvalid,

	#[error("Rate is NaN, zero or less than zero")]
	RateInvalid,

	#[error("Shape and rate are both infinite")]
	ShapeAndRateInfinite,
}

#[cfg(feature = "python")]
impl_pyerr!(GammaError, pyo3::exceptions::PyValueError);

impl Gamma {
	/// Constructs a new gamma distribution with a shape (α) of `shape` and
	/// a rate (β) of `rate`
	///
	/// Shape and rate must be positive non-zero non-NaN values.  Either
	/// shape or rate can be infinite, but if both of them are, an error is
	/// returned.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::{Gamma, GammaError};
	///
	/// assert!(Gamma::new(3.0, 1.0).is_ok());
	/// assert_eq!(Gamma::new(0.0, 1.0), Err(GammaError::ShapeInvalid));
	/// assert!(Gamma::new(f64::INFINITY, 2.0).is_ok());
	/// assert_eq!(
	///     Gamma::new(f64::INFINITY, f64::INFINITY),
	///     Err(GammaError::ShapeAndRateInfinite)
	/// );
	/// ```
	pub fn new(shape: f64, rate: f64) -> Result<Gamma, GammaError> {
		if shape.is_nan() || shape <= 0.0 {
			return Err(GammaError::ShapeInvalid);
		}

		if rate.is_nan() || rate <= 0.0 {
			return Err(GammaError::RateInvalid);
		}

		if shape.is_infinite() && rate.is_infinite() {
			return Err(GammaError::ShapeAndRateInfinite);
		}

		Ok(Gamma { shape, rate })
	}

	/// Returns the shape (α) of the gamma distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Gamma;
	///
	/// let g = Gamma::new(3.0, 1.0).unwrap();
	/// assert_eq!(g.shape(), 3.0);
	/// ```
	pub fn shape(&self) -> f64 {
		self.shape
	}

	/// Returns the rate (β) of the gamma distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::Gamma;
	///
	/// let g = Gamma::new(3.0, 1.0).unwrap();
	/// assert_eq!(g.rate(), 1.0);
	/// ```
	pub fn rate(&self) -> f64 {
		self.rate
	}
}

impl core::fmt::Display for Gamma {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Γ({}, {})", self.shape, self.rate)
	}
}

impl ContinuousCDF for Gamma {
	/// `(1 / Γ(α)) * γ(α, β * x)`, where `α` is the shape, `β` is the rate,
	/// `Γ` is the gamma function, and `γ` is the lower incomplete gamma
	/// function.
	fn cdf(&self, x: f64) -> f64 {
		if x <= 0.0 {
			0.0
		} else if ulps_eq!(x, self.shape) && self.rate.is_infinite() {
			1.0
		} else if self.rate.is_infinite() {
			0.0
		} else if x.is_infinite() {
			1.0
		} else {
			gamma::gamma_lr(self.shape, x * self.rate)
		}
	}

	/// `(1 / Γ(α)) * γ(α, β * x)` where `α` is the shape, `β` is the rate,
	/// `Γ` is the gamma function, and `γ` is the upper incomplete gamma
	/// function.
	fn sf(&self, x: f64) -> f64 {
		if x <= 0.0 {
			1.0
		} else if ulps_eq!(x, self.shape) && self.rate.is_infinite() {
			0.0
		} else if self.rate.is_infinite() {
			1.0
		} else if x.is_infinite() {
			0.0
		} else {
			gamma::gamma_ur(self.shape, x * self.rate)
		}
	}

	fn inverse_cdf(&self, p: f64) -> f64 {
		if !(0.0..=1.0).contains(&p) {
			panic!(
				"default inverse_cdf implementation should be provided probability on [0,1]"
			)
		}
		if p == 0.0 {
			return self.lower();
		};
		if p == 1.0 {
			return self.upper();
		};

		// Bisection search for MAX_ITERS.0 iterations
		let mut high = 2.0;
		let mut low = 1.0;
		while self.cdf(low) > p {
			low /= 2.0;
		}
		while self.cdf(high) < p {
			high *= 2.0;
		}
		let mut x_0 = (high + low) / 2.0;

		for _ in 0..8 {
			if self.cdf(x_0) >= p {
				high = x_0;
			} else {
				low = x_0;
			}
			if prec::convergence(&mut x_0, (high + low) / 2.0) {
				break;
			}
		}

		// Newton Raphson, for at least one step
		for _ in 0..4 {
			let x_next = x_0 - (self.cdf(x_0) - p) / self.pdf(x_0);
			if prec::convergence(&mut x_0, x_next) {
				break;
			}
		}

		x_0
	}

	fn lower(&self) -> f64 {
		0.0
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Gamma {
	/// `α / β`, where `α` is the shape and `β` is the rate
	fn mean(&self) -> Option<f64> {
		Some(self.shape / self.rate)
	}

	/// `α / β^2`, where `α` is the shape and `β` is the rate
	fn variance(&self) -> Option<f64> {
		Some(self.shape / (self.rate * self.rate))
	}

	/// `α - ln(β) + ln(Γ(α)) + (1 - α) * ψ(α)`, where `α` is the shape, `β`
	/// is the rate, `Γ` is the gamma function, and `ψ` is the digamma
	/// function.
	fn entropy(&self) -> Option<f64> {
		let entr = self.shape - self.rate.ln()
			+ gamma::ln_gamma(self.shape)
			+ (1.0 - self.shape) * gamma::digamma(self.shape);
		Some(entr)
	}

	/// `2 / sqrt(α)`, where `α` is the shape
	fn skewness(&self) -> Option<f64> {
		Some(2.0 / self.shape.sqrt())
	}
}

impl Mode<Option<f64>> for Gamma {
	/// `(α - 1) / β`, where `α` is the shape and `β` is the rate.
	///
	/// If `α < 1`, this method returns `None`.
	fn mode(&self) -> Option<f64> {
		if self.shape < 1.0 {
			None
		} else {
			Some((self.shape - 1.0) / self.rate)
		}
	}
}

impl Continuous for Gamma {
	type T = f64;

	/// `(β^α / Γ(α)) * x^(α - 1) * e^(-β * x)`, where `α` is the shape, `β`
	/// is the rate, and `Γ` is the gamma function.
	///
	/// Returns `NAN` if any of `shape` or `rate` are `f64::INFINITY` or if
	/// `x` is `f64::INFINITY`.
	fn pdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			0.0
		} else if ulps_eq!(self.shape, 1.0) {
			self.rate * (-self.rate * x).exp()
		} else if self.shape > 160.0 {
			self.ln_pdf(x).exp()
		} else if x.is_infinite() {
			0.0
		} else {
			self.rate.powf(self.shape)
				* x.powf(self.shape - 1.0) * (-self.rate * x).exp()
				/ gamma::gamma(self.shape)
		}
	}

	/// See [`pdf`][Gamma::pdf] for the notes on degenerate cases handling.
	fn ln_pdf(&self, x: f64) -> f64 {
		if x < 0.0 {
			f64::NEG_INFINITY
		} else if ulps_eq!(self.shape, 1.0) {
			self.rate.ln() - self.rate * x
		} else if x.is_infinite() {
			f64::NEG_INFINITY
		} else {
			self.shape * self.rate.ln()
				+ (self.shape - 1.0) * x.ln()
				- self.rate * x - gamma::ln_gamma(self.shape)
		}
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Gamma {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		sample_unchecked(rng, self.shape, self.rate)
	}
}

/// Samples from a gamma distribution with a shape of `shape` and a rate of
/// `rate` using `rng` as the source of randomness. Implementation from:
///
/// _"A Simple Method for Generating Gamma Variables"_ - Marsaglia & Tsang
///
/// ACM Transactions on Mathematical Software, Vol. 26, No. 3, September 2000,
/// Pages 363-372
#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
pub fn sample_unchecked<R: rand::Rng + ?Sized>(
	rng: &mut R,
	shape: f64,
	rate: f64,
) -> f64 {
	let mut a = shape;
	let mut afix = 1.0;
	if shape < 1.0 {
		a = shape + 1.0;
		afix = rng.random::<f64>().powf(1.0 / shape);
	}

	let d = a - 1.0 / 3.0;
	let c = 1.0 / (9.0 * d).sqrt();
	loop {
		let mut x;
		let mut v;
		loop {
			x = super::normal::sample_unchecked(rng, 0.0, 1.0);
			v = 1.0 + c * x;
			if v > 0.0 {
				break;
			};
		}

		v = v * v * v;
		x = x * x;
		let u: f64 = rng.random();
		if u < 1.0 - 0.0331 * x * x
			|| u.ln() < 0.5 * x + d * (1.0 - v + v.ln())
		{
			return afix * d * v / rate;
		}
	}
}
