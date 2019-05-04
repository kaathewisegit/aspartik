use distribution::{Continuous, Univariate};
use rand::distributions::Distribution;
use rand::distributions::OpenClosed01;
use rand::Rng;
use statistics::*;
use std::f64;
use {Result, StatsError};

/// Continuous Univariate Laplace distribution.
/// The Laplace distribution is a distribution over the real numbers parameterized by a mean and
/// scale parameter. The PDF is:
///     p(x) = \frac{1}{2 * scale} \exp{- |x - mean| / scale}.
/// <a href="http://en.wikipedia.org/wiki/Laplace_distribution">Wikipedia - Laplace distribution</a>.
pub struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
    fn sample_unchecked<R: Rng + ?Sized>(r: &mut R, location: f64, scale: f64) -> f64 {
        let r: f64 = r.gen();
        let u = r - 0.5;
        location - (scale * f64::signum(u) * f64::ln(1.0 - (2.0 * f64::abs(u))))
    }

    fn valid_parameter_set(location: f64, scale: f64) -> bool {
        scale > 0.0 && !location.is_nan()
    }

    /// Initializes a new instance of the Laplace struct.
    /// returns an error is scale is negative
    pub fn new(location: f64, scale: f64) -> Result<Laplace> {
        if Laplace::valid_parameter_set(location, scale) {
            Ok(Laplace { location, scale })
        } else {
            Err(StatsError::BadParams)
        }
    }
}

impl Mean<f64> for Laplace {
    /// Gets the mean of the distribution.
    fn mean(&self) -> f64 {
        self.location
    }
}

impl Variance<f64> for Laplace {
    /// Gets the variance of the distribution.
    fn variance(&self) -> f64 {
        2.0 * self.scale * self.scale
    }

    /// Gets the standard deviation of the distribution.
    fn std_dev(&self) -> f64 {
        f64::consts::SQRT_2 * self.scale
    }
}

impl Entropy<f64> for Laplace {
    /// Gets the entropy of the distribution.
    fn entropy(&self) -> f64 {
        f64::ln(2.0 * f64::consts::E * self.scale)
    }
}

impl Skewness<f64> for Laplace {
    /// Gets the skewness of the distribution.
    fn skewness(&self) -> f64 {
        0.0
    }
}

impl Mode<f64> for Laplace {
    /// Gets the mode of the distribution.
    fn mode(&self) -> f64 {
        self.location
    }
}

impl Median<f64> for Laplace {
    /// Gets the median of the distribution.
    fn median(&self) -> f64 {
        self.location
    }
}

impl Min<f64> for Laplace {
    /// Gets the minimum of the distribution.
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for Laplace {
    /// Gets the maximum of the distribution.
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl Continuous<f64, f64> for Laplace {
    /// Computes the probability density of the distribution (PDF) at x, i.e. ∂P(X ≤ x)/∂x.
    fn pdf(&self, x: f64) -> f64 {
        f64::exp(-f64::abs(x - self.location) / self.scale) / (2.0 * self.scale)
    }

    /// Computes the log probability density of the distribution (lnPDF) at x, i.e. ln(∂P(X ≤ x)/∂x).
    fn ln_pdf(&self, x: f64) -> f64 {
        -f64::abs(x - self.location) / self.scale - f64::ln(2.0 * self.scale)
    }
}

impl Distribution<f64> for Laplace {
    /// Samples a Laplace distributed random variable.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        Laplace::sample_unchecked(rng, self.location, self.scale)
    }
}

impl Univariate<f64, f64> for Laplace {
    /// Computes the cumulative distribution (CDF) of the distribution at x, i.e. P(X ≤ x).
    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0
            + (f64::signum(x - self.location)
                * (1.0 - f64::exp(-f64::abs(x - self.location) / self.scale))))
    }
}

#[cfg(test)]
mod test {
    use distribution::internal::*;
    use distribution::{laplace, Continuous, Univariate};
    use rand::distributions::Distribution;
    use rand::thread_rng;
    use statistics::*;
    use std::f64;

    fn try_create(location: f64, scale: f64) -> laplace::Laplace {
        let n = laplace::Laplace::new(location, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(location: f64, scale: f64) {
        let n = try_create(location, scale);
        assert_eq!(location, n.location);
        assert_eq!(scale, n.scale);
    }

    fn bad_create_case(location: f64, scale: f64) {
        let n = laplace::Laplace::new(location, scale);
        assert!(n.is_err());
    }

    fn test_case<F>(location: f64, scale: f64, expected: f64, eval: F)
    where
        F: Fn(laplace::Laplace) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_is_nan<F>(location: f64, scale: f64, eval: F)
    where
        F: Fn(laplace::Laplace) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert!(x.is_nan());
    }

    fn test_almost<F>(location: f64, scale: f64, expected: f64, acc: f64, eval: F)
    where
        F: Fn(laplace::Laplace) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        try_create(1.0, 2.0);
        try_create(f64::NEG_INFINITY, 0.1);
        try_create(-5.0 - 1.0, 1.0);
        try_create(0.0, 5.0);
        try_create(1.0, 7.0);
        try_create(5.0, 10.0);
        try_create(f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_create_fail() {
        bad_create_case(2.0, -1.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(f64::NAN, -1.0);
    }

    #[test]
    fn test_mean() {
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, |l| l.mean());
        test_case(-5.0 - 1.0, 1.0, -6.0, |l| l.mean());
        test_case(0.0, 5.0, 0.0, |l| l.mean());
        test_case(1.0, 10.0, 1.0, |l| l.mean());
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, |l| l.mean());
    }

    #[test]
    fn test_variance() {
        test_almost(f64::NEG_INFINITY, 0.1, 0.02, 1E-12, |l| l.variance());
        test_almost(-5.0 - 1.0, 1.0, 2.0, 1E-12, |l| l.variance());
        test_almost(0.0, 5.0, 50.0, 1E-12, |l| l.variance());
        test_almost(1.0, 7.0, 98.0, 1E-12, |l| l.variance());
        test_almost(5.0, 10.0, 200.0, 1E-12, |l| l.variance());
        test_almost(f64::INFINITY, f64::INFINITY, f64::INFINITY, 1E-12, |l| {
            l.variance()
        });
    }

    #[test]
    fn test_stddev() {
        test_almost(
            f64::NEG_INFINITY,
            0.1,
            f64::consts::SQRT_2 * 0.1,
            1E-12,
            |l| l.std_dev(),
        );
        test_almost(-5.0 - 1.0, 1.0, f64::consts::SQRT_2, 1E-12, |l| l.std_dev());
        test_almost(0.0, 5.0, f64::sqrt(50.0), 1E-12, |l| l.std_dev());
        test_almost(1.0, 7.0, f64::sqrt(98.0), 1E-12, |l| l.std_dev());
        test_almost(5.0, 10.0, f64::sqrt(200.0), 1E-12, |l| l.std_dev());
        test_almost(f64::INFINITY, f64::INFINITY, f64::INFINITY, 1E-12, |l| {
            l.std_dev()
        });
    }

    #[test]
    fn test_entropy() {
        test_almost(
            f64::NEG_INFINITY,
            0.1,
            f64::ln(2.0 * f64::consts::E * 0.1),
            1E-12,
            |l| l.entropy(),
        );
        test_almost(-6.0, 1.0, f64::ln(2.0 * f64::consts::E), 1E-12, |l| {
            l.entropy()
        });
        test_almost(1.0, 7.0, f64::ln(2.0 * f64::consts::E * 7.0), 1E-12, |l| {
            l.entropy()
        });
        test_almost(
            5.0,
            10.0,
            f64::ln(2.0 * f64::consts::E * 10.0),
            1E-12,
            |l| l.entropy(),
        );
        test_almost(f64::INFINITY, f64::INFINITY, f64::INFINITY, 1E-12, |l| {
            l.entropy()
        });
    }

    #[test]
    fn test_skewness() {
        test_case(f64::NEG_INFINITY, 0.1, 0.0, |l| l.skewness());
        test_case(-6.0, 1.0, 0.0, |l| l.skewness());
        test_case(1.0, 7.0, 0.0, |l| l.skewness());
        test_case(5.0, 10.0, 0.0, |l| l.skewness());
        test_case(f64::INFINITY, f64::INFINITY, 0.0, |l| l.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, |l| l.mode());
        test_case(-6.0, 1.0, -6.0, |l| l.mode());
        test_case(1.0, 7.0, 1.0, |l| l.mode());
        test_case(5.0, 10.0, 5.0, |l| l.mode());
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, |l| l.mode());
    }

    #[test]
    fn test_median() {
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, |l| l.median());
        test_case(-6.0, 1.0, -6.0, |l| l.median());
        test_case(1.0, 7.0, 1.0, |l| l.median());
        test_case(5.0, 10.0, 5.0, |l| l.median());
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, |l| l.median());
    }

    #[test]
    fn test_min() {
        test_case(0.0, 1.0, f64::NEG_INFINITY, |l| l.min());
    }

    #[test]
    fn test_max() {
        test_case(0.0, 1.0, f64::INFINITY, |l| l.max());
    }

    #[test]
    fn test_density() {
        test_almost(0.0, 0.1, 1.529511602509129e-06, 1E-12, |l| l.pdf(1.5));
        test_almost(1.0, 0.1, 7.614989872356341e-08, 1E-12, |l| l.pdf(2.8));
        test_almost(-1.0, 0.1, 3.8905661205668983e-19, 1E-12, |l| l.pdf(-5.4));
        test_almost(5.0, 0.1, 5.056107463052243e-43, 1E-12, |l| l.pdf(-4.9));
        test_almost(-5.0, 0.1, 1.9877248679543235e-30, 1E-12, |l| l.pdf(2.0));
        test_almost(f64::INFINITY, 0.1, 0.0, 1E-12, |l| l.pdf(5.5));
        test_almost(f64::NEG_INFINITY, 0.1, 0.0, 1E-12, |l| l.pdf(-0.0));
        test_almost(0.0, 1.0, 0.0, 1E-12, |l| l.pdf(f64::INFINITY));
        test_almost(1.0, 1.0, 0.00915781944436709, 1E-12, |l| l.pdf(5.0));
        test_almost(-1.0, 1.0, 0.5, 1E-12, |l| l.pdf(-1.0));
        test_almost(5.0, 1.0, 0.0012393760883331792, 1E-12, |l| l.pdf(-1.0));
        test_almost(-5.0, 1.0, 0.0002765421850739168, 1E-12, |l| l.pdf(2.5));
        test_almost(f64::INFINITY, 0.1, 0.0, 1E-12, |l| l.pdf(2.0));
        test_almost(f64::NEG_INFINITY, 0.1, 0.0, 1E-12, |l| l.pdf(15.0));
        test_almost(0.0, f64::INFINITY, 0.0, 1E-12, |l| l.pdf(89.3));
        test_almost(1.0, f64::INFINITY, 0.0, 1E-12, |l| l.pdf(-0.1));
        test_almost(-1.0, f64::INFINITY, 0.0, 1E-12, |l| l.pdf(0.1));
        test_almost(5.0, f64::INFINITY, 0.0, 1E-12, |l| l.pdf(-6.1));
        test_almost(-5.0, f64::INFINITY, 0.0, 1E-12, |l| l.pdf(-10.0));
        test_is_nan(f64::INFINITY, f64::INFINITY, |l| l.pdf(2.0));
        test_is_nan(f64::NEG_INFINITY, f64::INFINITY, |l| l.pdf(-5.1));
    }

    #[test]
    fn test_ln_density() {
        test_almost(0.0, 0.1, -13.3905620875659, 1E-12, |l| l.ln_pdf(1.5));
        test_almost(1.0, 0.1, -16.390562087565897, 1E-12, |l| l.ln_pdf(2.8));
        test_almost(-1.0, 0.1, -42.39056208756591, 1E-12, |l| l.ln_pdf(-5.4));
        test_almost(5.0, 0.1, -97.3905620875659, 1E-12, |l| l.ln_pdf(-4.9));
        test_almost(-5.0, 0.1, -68.3905620875659, 1E-12, |l| l.ln_pdf(2.0));
        test_case(f64::INFINITY, 0.1, f64::NEG_INFINITY, |l| l.ln_pdf(5.5));
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, |l| {
            l.ln_pdf(-0.0)
        });
        test_case(0.0, 1.0, f64::NEG_INFINITY, |l| l.ln_pdf(f64::INFINITY));
        test_almost(1.0, 1.0, -4.693147180559945, 1E-12, |l| l.ln_pdf(5.0));
        test_almost(-1.0, 1.0, -f64::consts::LN_2, 1E-12, |l| l.ln_pdf(-1.0));
        test_almost(5.0, 1.0, -6.693147180559945, 1E-12, |l| l.ln_pdf(-1.0));
        test_almost(-5.0, 1.0, -8.193147180559945, 1E-12, |l| l.ln_pdf(2.5));
        test_case(f64::INFINITY, 0.1, f64::NEG_INFINITY, |l| l.ln_pdf(2.0));
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, |l| {
            l.ln_pdf(15.0)
        });
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |l| l.ln_pdf(89.3));
        test_case(1.0, f64::INFINITY, f64::NEG_INFINITY, |l| l.ln_pdf(-0.1));
        test_case(-1.0, f64::INFINITY, f64::NEG_INFINITY, |l| l.ln_pdf(0.1));
        test_case(5.0, f64::INFINITY, f64::NEG_INFINITY, |l| l.ln_pdf(-6.1));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, |l| l.ln_pdf(-10.0));
        test_is_nan(f64::INFINITY, f64::INFINITY, |l| l.ln_pdf(2.0));
        test_is_nan(f64::NEG_INFINITY, f64::INFINITY, |l| l.ln_pdf(-5.1));
    }

    #[test]
    fn test_sample() {
        let l = try_create(0.1, 0.5);
        l.sample(&mut thread_rng());
    }
}
