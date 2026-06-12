use computare_special::gamma;

use core::{f64::consts, num::NonZeroU64};

use crate::{
	distribution::{Continuous, ContinuousCDF},
	statistics::{Distribution, Mode},
};

/// [Chi distribution](https://en.wikipedia.org/wiki/Chi_distribution)
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Chi {
	freedom: NonZeroU64,
}

impl Chi {
	/// Constructs a new chi distribution with `freedom` degrees of freedom
	pub fn new(freedom: NonZeroU64) -> Chi {
		Self { freedom }
	}

	/// The degrees of freedom of the chi distribution.
	///
	/// Never zero.  Use the `freedom_nz` method to get `NonZeroU64`.
	pub fn freedom(&self) -> u64 {
		self.freedom.get()
	}

	pub fn freedom_nz(&self) -> NonZeroU64 {
		self.freedom
	}
}

impl core::fmt::Display for Chi {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "Chi({})", self.freedom)
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl rand::distr::Distribution<f64> for Chi {
	fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
		(0..self.freedom())
			.fold(0.0, |acc, _| {
				acc + super::normal::sample_unchecked(
					rng, 0.0, 1.0,
				)
				.powf(2.0)
			})
			.sqrt()
	}
}

impl ContinuousCDF for Chi {
	/// `P(k / 2, x^2 / 2)`
	///
	/// `P` is the regularized lower incomplete Gamma function.
	fn cdf(&self, x: f64) -> f64 {
		if x == f64::INFINITY {
			1.0
		} else if x <= 0.0 {
			0.0
		} else {
			gamma::regularized_lower_gamma(
				self.freedom() as f64 / 2.0,
				x * x / 2.0,
			)
		}
	}

	/// `P(k / 2, x^2 / 2)`
	///
	/// Where `P` is the regularized upper incomplete Gamma function.
	fn sf(&self, x: f64) -> f64 {
		if x == f64::INFINITY {
			0.0
		} else if x <= 0.0 {
			1.0
		} else {
			gamma::regularized_upper_gamma(
				self.freedom() as f64 / 2.0,
				x * x / 2.0,
			)
		}
	}

	/// Always 0
	fn lower(&self) -> f64 {
		0.0
	}

	/// Always infinity
	fn upper(&self) -> f64 {
		f64::INFINITY
	}
}

impl Distribution for Chi {
	/// `sqrt2 * Γ((k + 1) / 2) / Γ(k / 2)`
	fn mean(&self) -> Option<f64> {
		let freedom = self.freedom() as f64;

		if self.freedom() > 300 {
			// Large n approximation based on the Stirling series approximation to the Gamma function
			// This avoids call the Gamma function with large arguments and returning NaN
			//
			// Relative accuracy follows O(1/n^4) and at 300 d.o.f. is better than 1e-12
			// For a f32 impl the threshold should be changed to 150
			Some((freedom.sqrt())
				/ ((1.0 + 0.25 / freedom)
					* (1.0 + 0.03125
						/ (freedom * freedom)) * (1.0 - 0.046875
					/ (freedom * freedom * freedom))))
		} else {
			let mean = consts::SQRT_2
				* gamma::gamma((freedom + 1.0) / 2.0)
				/ gamma::gamma(freedom / 2.0);
			Some(mean)
		}
	}

	/// `k - μ^2`, where `μ` is the [mean]
	///
	/// [mean]: [Self::mean]
	fn variance(&self) -> Option<f64> {
		let mean = self.mean()?;
		Some(self.freedom() as f64 - mean * mean)
	}

	/// `ln(Γ(k / 2)) + 0.5 * (k - ln2 - (k - 1) * ψ(k / 2))`
	///
	/// Where `k` is degrees of freedom, `Γ` is the gamma function, and `ψ`
	/// is the digamma function.
	fn entropy(&self) -> Option<f64> {
		let freedom = self.freedom() as f64;
		let entr = gamma::ln_gamma(freedom / 2.0)
			+ (freedom
				- (2.0f64).ln() - (freedom - 1.0)
				* gamma::digamma(freedom / 2.0))
				/ 2.0;
		Some(entr)
	}

	/// `(μ / σ^3) * (1 - 2σ^2)`
	///
	/// Where `μ` is the mean and `σ` the standard deviation of the
	/// distribution.
	fn skewness(&self) -> Option<f64> {
		let sigma = self.std_dev()?;
		let skew = self.mean()? * (1.0 - 2.0 * sigma * sigma)
			/ (sigma * sigma * sigma);
		Some(skew)
	}
}

impl Mode<Option<f64>> for Chi {
	/// `sqrt(k - 1)`
	fn mode(&self) -> Option<f64> {
		Some(((self.freedom() - 1) as f64).sqrt())
	}
}

impl Continuous for Chi {
	/// `(2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2)`
	///
	/// Where `k` is the degrees of freedom and `Γ` is the gamma function.
	fn pdf(&self, x: f64) -> f64 {
		if x == f64::INFINITY || x <= 0.0 {
			0.0
		} else if self.freedom() > 160 {
			self.ln_pdf(x).exp()
		} else {
			let freedom = self.freedom() as f64;
			(2.0f64).powf(1.0 - freedom / 2.0)
				* x.powf(freedom - 1.0) * (-x * x / 2.0).exp()
				/ gamma::gamma(freedom / 2.0)
		}
	}

	/// `ln((2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2))`
	fn ln_pdf(&self, x: f64) -> f64 {
		if x == f64::INFINITY || x <= 0.0 {
			f64::NEG_INFINITY
		} else {
			let freedom = self.freedom() as f64;
			(1.0 - freedom / 2.0) * (2.0f64).ln()
				+ ((freedom - 1.0) * x.ln()) - x * x / 2.0
				- gamma::ln_gamma(freedom / 2.0)
		}
	}
}
