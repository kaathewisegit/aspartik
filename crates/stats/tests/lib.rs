use approx::AbsDiffEq;
use num_traits::Float;
use stats::assert_almost_eq;
use stats::distribution::{Continuous, ContinuousCDF, Discrete, DiscreteCDF};

mod distributions;

/// Targeted accuracy instantiated over `f64`
pub const ACCURACY: f64 = 10e-11;

pub fn almost_eq(a: f64, b: f64, acc: f64) -> bool {
	if a.is_infinite() && b.is_infinite() {
		return a == b;
	}
	a.abs_diff_eq(&b, acc)
}

/// Compares if two floats are close via `approx::abs_diff_eq` using a maximum
/// absolute difference (epsilon) of `acc`.
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $prec:expr $(,)?) => {
        if !$crate::almost_eq($a, $b, $prec) {
            panic!(
                "assertion failed: `abs(left - right) < {:e}`, (left: `{}`, right: `{}`)",
                $prec, $a, $b
            );
        }
    };
}

#[macro_export]
macro_rules! test_new_is_ok {
	($dist:ty; $( ($($arg:expr),+) ),+ $(,)?) => {
		#[test]
		fn test_new_is_ok() {
		$(
			let result = <$dist>::new($($arg),+);
			if !result.is_ok() {
				panic!(
					"Expected {}::new{:?} to return Ok(_), got {:?} instead",
					stringify!($dist),
					// args tuple
					($($arg),+),
					result,
				);
			}
		)+
		}
	};
}

#[macro_export]
macro_rules! test_new_is_err {
	($dist:ty; $( ($($arg:expr),+) -> $err:expr ),+ $(,)?) => {
		#[test]
		fn test_new_is_err() {
		$(
			let result = <$dist>::new($($arg),+);
			if result != Err($err) {
				panic!(
					"Expected {}::new{:?} to return Err({}), got {:?} instead",
					stringify!($dist),
					// args tuple
					($($arg),+),
					stringify!($err),
					result,
				);
			}
		)+
		}
	};
}

#[macro_export]
macro_rules! test_value {
	(
		$name:ident, $dist:ty, $func:tt;
	 	$(
			($($new_arg:expr),+):
			($($func_arg:expr),*) => $expected:expr
			$(, $mode:ident $($value:literal)? $($unwrap:ident)?)?
		),+
		$(,)?
	) => {
		#[test]
		fn $name() {
		$(
			let dist = <$dist>::new($($new_arg),+).unwrap();
			let result = dist.$func($($func_arg),*);
			if !$crate::compare!(
				result,
				$expected
				$(, $mode $($value)? $($unwrap)?)?
			) {
				panic!(
					"Expected {}.{}({:?}) to return {:?}, got {:?} instead",
					dist,
					stringify!($func),
					($($func_arg),*),
					$expected,
					result,
				);
			}
		)+
		}
	};
}

pub fn is_infinite<F: Float>(f: F) -> bool {
	f.is_infinite()
}

#[macro_export]
macro_rules! compare {
	($result:expr, $expected:expr, $mode:ident $($value:literal)? unwrap) => {
		compare!($result.unwrap(), $expected, $mode $($value)?)
	};
	($result:expr, $expected:expr) => {
		$result == $expected
	};
	($result:expr, $expected:expr, relative) => {
		compare!($result, $expected, relative ACCURACY)
	};
	($result:expr, $expected:expr, relative $max_relative:expr) => {
		approx::relative_eq!(
			$expected,
			$result,
			max_relative = $max_relative
		)
	};
	($result:expr, $expected:expr, absolute) => {
		compare!($result, $expected, absolute 1e-16)
	};
	($result:expr, $expected:expr, absolute $epsilon:expr) => {
		// abs_diff_eq! cannot handle infinities, so we manually accept
		// them here
		(is_infinite($expected) && $result == $expected)
		||
		approx::abs_diff_eq!($expected, $result, epsilon = $epsilon)
	};
}

pub mod prelude {
	pub use super::{is_infinite, ACCURACY};
	pub use approx::{abs_diff_eq, AbsDiffEq};
	pub use num_traits::Float;
	pub use stats::{
		assert_almost_eq,
		distribution::{
			Continuous, ContinuousCDF, Discrete, DiscreteCDF,
		},
		statistics::{Distribution, Mode},
	};

	pub use super::{compare, test_new_is_err, test_new_is_ok, test_value};
}

mod macro_test {
	use stats::distribution::{Beta, BetaError};
	use stats::statistics::*;

	use crate::prelude::*;

	test_new_is_ok! {
		Beta;
		(0.8, 1.2),
		(10e10, 10e10),
	}

	test_new_is_err! {
		Beta;
		(-0.5, 1.2) -> BetaError::InvalidAlpha,
		(0.5, -0.0) -> BetaError::InvalidBeta,
	}

	test_value! {
		test_exact_success, Beta, mode;
		(1.5, 1.5): () => Some(0.5),
	}

	test_value! {
		test_relative_success, Beta, mode;
		(1.2, 1.4): () => 0.333333333333, relative unwrap,
	}

	test_value! {
		test_absolute_success, Beta, mode;
		(1.2, 1.4): () => 0.333333333333, absolute 1e-12 unwrap,
	}

	test_value! {
		test_is_none, Beta, mode;
		(0.5, 1.2): () => None::<f64>,
	}
}

/// cdf should be the integral of the pdf
fn check_integrate_pdf_is_cdf<D>(dist: &D, x_min: f64, x_max: f64, step: f64)
where
	D: ContinuousCDF + Continuous<T = f64>,
{
	let mut prev_x = x_min;
	let mut prev_density = dist.pdf(x_min);
	let mut sum = 0.0;

	loop {
		let x = prev_x + step;
		let density = dist.pdf(x);

		assert!(density >= 0.0);

		let ln_density = dist.ln_pdf(x);

		assert_almost_eq!(density.ln(), ln_density, 1e-10);

		// triangle rule
		sum += (prev_density + density) * step / 2.0;

		let cdf = dist.cdf(x);
		if (sum - cdf).abs() > 1e-3 {
			panic!("Integral of pdf doesn't equal cdf!\n\
                        Integration from {x_min} by {step} to {x} = {sum}\n\
                        cdf = {cdf}");
		}

		if x >= x_max {
			break;
		} else {
			prev_x = x;
			prev_density = density;
		}
	}

	assert!(sum > 0.99);
	assert!(sum <= 1.001);
}

/// cdf should be the sum of the pmf
fn check_sum_pmf_is_cdf<D>(dist: &D, x_max: u64)
where
	D: DiscreteCDF + Discrete<T = u64>,
{
	let mut sum = 0.0;

	// go slightly beyond x_max to test for off-by-one errors
	for i in 0..x_max + 3 {
		let prob = dist.pmf(i);

		assert!(prob >= 0.0);
		assert!(prob <= 1.0);

		sum += prob;

		if i == x_max {
			assert!(sum > 0.99);
		}

		assert_almost_eq!(sum, dist.cdf(i), 1e-10);
		// assert_almost_eq!(sum, dist.cdf(i as f64), 1e-10);
		// assert_almost_eq!(sum, dist.cdf(i as f64 + 0.1), 1e-10);
		// assert_almost_eq!(sum, dist.cdf(i as f64 + 0.5), 1e-10);
		// assert_almost_eq!(sum, dist.cdf(i as f64 + 0.9), 1e-10);
	}

	assert!(sum > 0.99);
	assert!(sum <= 1.0 + 1e-10);
}

/// pdf should be derivative of cdf
fn check_derivative_of_cdf_is_pdf<D>(
	dist: &D,
	x_min: f64,
	x_max: f64,
	step: f64,
) where
	D: ContinuousCDF + Continuous<T = f64>,
{
	const DELTA: f64 = 1e-12;
	const DX: f64 = 2.0 * DELTA;
	let mut prev_x = x_min;

	loop {
		let x = prev_x + step;
		let x_ahead = x + DELTA;
		let x_behind = x - DELTA;
		let density = dist.pdf(x);

		let d_cdf = dist.cdf(x_ahead) - dist.cdf(x_behind);

		assert_almost_eq!(d_cdf, DX * density, 1e-11);

		if x >= x_max {
			break;
		} else {
			prev_x = x;
		}
	}
}

/// Does a series of checks that all continuous distributions must obey.  99% of
/// the probability mass should be between x_min and x_max or the finite
/// difference of cdf should be near to the pdf for much of the support.
pub fn check_continuous_distribution<D>(dist: &D, x_min: f64, x_max: f64)
where
	D: ContinuousCDF + Continuous<T = f64> + std::panic::RefUnwindSafe,
{
	assert_eq!(dist.pdf(f64::NEG_INFINITY), 0.0);
	assert_eq!(dist.pdf(f64::INFINITY), 0.0);
	assert_eq!(dist.ln_pdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
	assert_eq!(dist.ln_pdf(f64::INFINITY), f64::NEG_INFINITY);
	assert_eq!(dist.cdf(f64::NEG_INFINITY), 0.0);
	assert_eq!(dist.cdf(f64::INFINITY), 1.0);

	if std::panic::catch_unwind(|| {
		check_integrate_pdf_is_cdf(
			dist,
			x_min,
			x_max,
			(x_max - x_min) / 100000.0,
		);
	})
	.or(std::panic::catch_unwind(|| {
		check_derivative_of_cdf_is_pdf(
			dist,
			x_min,
			x_max,
			(x_max - x_min) / 100000.0,
		);
	}))
	.is_err()
	{
		panic!("Integration of pdf doesn't equal cdf and derivative of cdf doesn't equal pdf!");
	}
}

/// Does a series of checks that all positive discrete distributions must obey.
/// 99% of the probability mass should be between 0 and x_max (inclusive).
pub fn check_discrete_distribution<D>(dist: &D, x_max: u64)
where
	D: DiscreteCDF + Discrete<T = u64>,
{
	// assert_eq!(dist.cdf(f64::NEG_INFINITY), 0.0);
	// assert_eq!(dist.cdf(-10.0), 0.0);
	// assert_eq!(dist.cdf(-1.0), 0.0);
	// assert_eq!(dist.cdf(-0.01), 0.0);
	// assert_eq!(dist.cdf(f64::INFINITY), 1.0);

	check_sum_pmf_is_cdf(dist, x_max);
}
