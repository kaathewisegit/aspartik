use math::function::{beta, gamma};

use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};
use core::f64;

/// Implements the [Student's
/// T](https://en.wikipedia.org/wiki/Student%27s_t-distribution) distribution
///
/// # Examples
///
/// ```
/// use stats::distribution::{StudentsT, Continuous};
/// use stats::statistics::Distribution;
/// use math::assert_almost_eq;
///
/// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.0);
/// assert_almost_eq!(n.pdf(0.0), 0.353553390593274, epsilon = 1e-15);
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct StudentsT {
	location: f64,
	scale: f64,
	freedom: f64,
}

/// Represents the errors that can occur when creating a [`StudentsT`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum StudentsTError {
	/// The location is NaN.
	LocationInvalid,

	/// The scale is NaN, zero or less than zero.
	ScaleInvalid,

	/// The degrees of freedom are NaN, zero or less than zero.
	FreedomInvalid,
}

impl core::fmt::Display for StudentsTError {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			StudentsTError::LocationInvalid => {
				write!(f, "Location is NaN")
			}
			StudentsTError::ScaleInvalid => write!(
				f,
				"Scale is NaN, zero or less than zero"
			),
			StudentsTError::FreedomInvalid => {
				write!(
					f,
					"Degrees of freedom are NaN, zero or less than zero"
				)
			}
		}
	}
}

impl std::error::Error for StudentsTError {}

impl StudentsT {
	/// Constructs a new student's t-distribution with location `location`,
	/// scale `scale`, and `freedom` freedom.
	///
	/// # Errors
	///
	/// Returns an error if any of `location`, `scale`, or `freedom` are `NaN`.
	/// Returns an error if `scale <= 0.0` or `freedom <= 0.0`.
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::StudentsT;
	///
	/// let mut result = StudentsT::new(0.0, 1.0, 2.0);
	/// assert!(result.is_ok());
	///
	/// result = StudentsT::new(0.0, 0.0, 0.0);
	/// assert!(result.is_err());
	/// ```
	pub fn new(
		location: f64,
		scale: f64,
		freedom: f64,
	) -> Result<StudentsT, StudentsTError> {
		if location.is_nan() {
			return Err(StudentsTError::LocationInvalid);
		}

		if scale.is_nan() || scale <= 0.0 {
			return Err(StudentsTError::ScaleInvalid);
		}

		if freedom.is_nan() || freedom <= 0.0 {
			return Err(StudentsTError::FreedomInvalid);
		}

		Ok(StudentsT {
			location,
			scale,
			freedom,
		})
	}

	/// Returns the location of the student's t-distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::StudentsT;
	///
	/// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
	/// assert_eq!(n.location(), 0.0);
	/// ```
	pub fn location(&self) -> f64 {
		self.location
	}

	/// Returns the scale of the student's t-distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::StudentsT;
	///
	/// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
	/// assert_eq!(n.scale(), 1.0);
	/// ```
	pub fn scale(&self) -> f64 {
		self.scale
	}

	/// Returns the freedom of the student's t-distribution
	///
	/// # Examples
	///
	/// ```
	/// use stats::distribution::StudentsT;
	///
	/// let n = StudentsT::new(0.0, 1.0, 2.0).unwrap();
	/// assert_eq!(n.freedom(), 2.0);
	/// ```
	pub fn freedom(&self) -> f64 {
		self.freedom
	}
}

impl core::fmt::Display for StudentsT {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(
			f,
			"t_{}({},{})",
			self.freedom, self.location, self.scale
		)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for StudentsT {
	fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
		// based on method 2, section 5 in chapter 9 of L. Devroye's
		// "Non-Uniform Random Variate Generation"
		let gamma = super::gamma::sample_unchecked(
			r,
			0.5 * self.freedom,
			0.5,
		);
		super::normal::sample_unchecked(
			r,
			self.location,
			self.scale * (self.freedom / gamma).sqrt(),
		)
	}
}

impl ContinuousCDF for StudentsT {
	/// Calculates the cumulative distribution function for the student's
	/// t-distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// if x < μ {
	///     (1 / 2) * I(t, v / 2, 1 / 2)
	/// } else {
	///     1 - (1 / 2) * I(t, v / 2, 1 / 2)
	/// }
	/// ```
	///
	/// where `t = v / (v + k^2)`, `k = (x - μ) / σ`, `μ` is the location,
	/// `σ` is the scale, `v` is the freedom, and `I` is the regularized
	/// incomplete beta function
	fn cdf(&self, x: f64) -> f64 {
		if self.freedom.is_infinite() {
			super::normal::cdf_unchecked(
				x,
				self.location,
				self.scale,
			)
		} else {
			let k = (x - self.location) / self.scale;
			let h = self.freedom / (self.freedom + k * k);
			let ib = 0.5 * beta::beta_reg(
				self.freedom / 2.0,
				0.5,
				h,
			)
			// panics?
			.unwrap();
			if x <= self.location { ib } else { 1.0 - ib }
		}
	}

	/// Calculates the survival function for the student's t-distribution at
	/// `x`.
	///
	/// # Formula
	///
	/// ```text
	/// if x < μ {
	///     1 - (1 / 2) * I(t, v / 2, 1 / 2)
	/// } else {
	///     (1 / 2) * I(t, v / 2, 1 / 2)
	/// }
	/// ```
	///
	/// where `t = v / (v + k^2)`, `k = (x - μ) / σ`, `μ` is the location,
	/// `σ` is the scale, `v` is the freedom, and `I` is the regularized
	/// incomplete beta function
	fn sf(&self, x: f64) -> f64 {
		if self.freedom.is_infinite() {
			super::normal::sf_unchecked(
				x,
				self.location,
				self.scale,
			)
		} else {
			let k = (x - self.location) / self.scale;
			let h = self.freedom / (self.freedom + k * k);
			let ib = 0.5 * beta::beta_reg(
				self.freedom / 2.0,
				0.5,
				h,
			)
			// XXX: panics?
			.unwrap();
			if x <= self.location { 1.0 - ib } else { ib }
		}
	}

	/// Calculates the inverse cumulative distribution function for the
	/// Student's T-distribution at `x`
	fn inverse_cdf(&self, x: f64) -> f64 {
		// first calculate inverse_cdf for normal Student's T
		assert!((0.0..=1.0).contains(&x));
		let x1 = if x >= 0.5 { 1.0 - x } else { x };
		let a = 0.5 * self.freedom;
		let b = 0.5;
		let mut y = beta::inv_beta_reg(a, b, 2.0 * x1);
		y = (self.freedom * (1.0 - y) / y).sqrt();
		y = if x >= 0.5 { y } else { -y };
		// generalised Student's T is related to normal Student's T by `Y = μ + σ X`
		// where `X` is distributed as Student's T, so this result has to be scaled and shifted back
		// formally: F_Y(t) = P(Y <= t) = P(X <= (t - μ) / σ) = F_X((t - μ) / σ)
		// F_Y^{-1}(p) = inf { t' | F_Y(t') >= p } = inf { t' = μ + σ t | F_X((t' - μ) / σ) >= p }
		// because scale is positive: loc + scale * t is strictly monotonic function
		// = μ + σ inf { t | F_X(t) >= p } = μ + σ F_X^{-1}(p)
		self.location + self.scale * y
	}

	fn lower(&self) -> f64 {
		f64::NEG_INFINITY
	}

	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for StudentsT {
	/// Returns the mean of the student's t-distribution
	///
	/// # None
	///
	/// If `freedom <= 1.0`
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the location
	fn mean(&self) -> Option<f64> {
		if self.freedom <= 1.0 {
			None
		} else {
			Some(self.location)
		}
	}

	/// Returns the median of the student's t-distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the location
	fn median(&self) -> Option<f64> {
		Some(self.location)
	}

	/// Returns the variance of the student's t-distribution
	///
	/// # None
	///
	/// If `freedom <= 2.0`
	///
	/// # Formula
	///
	/// ```text
	/// if v == f64::INFINITY {
	///     Some(σ^2)
	/// } else if freedom > 2.0 {
	///     Some(v * σ^2 / (v - 2))
	/// } else {
	///     None
	/// }
	/// ```
	///
	/// where `σ` is the scale and `v` is the freedom
	fn variance(&self) -> Option<f64> {
		if self.freedom.is_infinite() {
			Some(self.scale * self.scale)
		} else if self.freedom > 2.0 {
			Some(self.freedom * self.scale * self.scale
				/ (self.freedom - 2.0))
		} else {
			None
		}
	}

	/// Returns the entropy for the student's t-distribution
	///
	/// # Formula
	///
	/// ```text
	/// - ln(σ) + (v + 1) / 2 * (ψ((v + 1) / 2) - ψ(v / 2)) + ln(sqrt(v) * B(v / 2, 1 /
	/// 2))
	/// ```
	///
	/// where `σ` is the scale, `v` is the freedom, `ψ` is the digamma function, and `B` is the
	/// beta function
	fn entropy(&self) -> Option<f64> {
		// generalised Student's T is related to normal Student's T by `Y = μ + σ X`
		// where `X` is distributed as Student's T, plugging into the definition
		// of entropy shows scaling affects the entropy by an additive constant `- ln σ`
		let shift = -self.scale.ln();
		let result = (self.freedom + 1.0) / 2.0
			* (gamma::digamma((self.freedom + 1.0) / 2.0)
				- gamma::digamma(self.freedom / 2.0))
			+ (self.freedom.sqrt()
				* beta::beta(self.freedom / 2.0, 0.5)
					// XXX: panics?
					.unwrap())
			.ln();
		Some(result + shift)
	}

	/// Returns the skewness of the student's t-distribution
	///
	/// # None
	///
	/// If `x <= 3.0`
	///
	/// # Formula
	///
	/// ```text
	/// 0
	/// ```
	fn skewness(&self) -> Option<f64> {
		if self.freedom <= 3.0 { None } else { Some(0.0) }
	}
}

impl Mode<Option<f64>> for StudentsT {
	/// Returns the mode of the student's t-distribution
	///
	/// # Formula
	///
	/// ```text
	/// μ
	/// ```
	///
	/// where `μ` is the location
	fn mode(&self) -> Option<f64> {
		Some(self.location)
	}
}

impl Continuous for StudentsT {
	type T = f64;

	/// Calculates the probability density function for the student's
	/// t-distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// Γ((v + 1) / 2) / (sqrt(vπ) * Γ(v / 2) * σ) * (1 + k^2 / v)^(-1 / 2 * (v
	/// + 1))
	/// ```
	///
	/// where `k = (x - μ) / σ`, `μ` is the location, `σ` is the scale, `v` is
	/// the freedom,
	/// and `Γ` is the gamma function
	fn pdf(&self, x: f64) -> f64 {
		if x.is_infinite() {
			0.0
		} else if self.freedom >= 1e8 {
			super::normal::pdf_unchecked(
				x,
				self.location,
				self.scale,
			)
		} else {
			let d = (x - self.location) / self.scale;
			(gamma::ln_gamma((self.freedom + 1.0) / 2.0)
				- gamma::ln_gamma(self.freedom / 2.0))
			.exp() * (1.0 + d * d / self.freedom)
				.powf(-0.5 * (self.freedom + 1.0))
				/ (self.freedom * f64::consts::PI).sqrt()
				/ self.scale
		}
	}

	/// Calculates the log probability density function for the student's
	/// t-distribution
	/// at `x`
	///
	/// # Formula
	///
	/// ```text
	/// ln(Γ((v + 1) / 2) / (sqrt(vπ) * Γ(v / 2) * σ) * (1 + k^2 / v)^(-1 / 2 *
	/// (v + 1)))
	/// ```
	///
	/// where `k = (x - μ) / σ`, `μ` is the location, `σ` is the scale, `v` is
	/// the freedom,
	/// and `Γ` is the gamma function
	fn ln_pdf(&self, x: f64) -> f64 {
		if x.is_infinite() {
			f64::NEG_INFINITY
		} else if self.freedom >= 1e8 {
			super::normal::ln_pdf_unchecked(
				x,
				self.location,
				self.scale,
			)
		} else {
			let d = (x - self.location) / self.scale;
			gamma::ln_gamma((self.freedom + 1.0) / 2.0)
				- 0.5 * ((self.freedom + 1.0)
					* (1.0 + d * d / self.freedom).ln())
				- gamma::ln_gamma(self.freedom / 2.0)
				- 0.5 * (self.freedom * f64::consts::PI).ln()
				- self.scale.ln()
		}
	}
}
