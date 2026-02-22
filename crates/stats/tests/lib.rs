use math::assert_almost_eq;

use stats::distribution::{Continuous, ContinuousCDF, Discrete, DiscreteCDF};

mod distributions;

/// NumPy's `assert_allclose` has the default relative accuracy of 1e-7.  SciPy
/// uses `sqrt(eps) * 4`, which is roughly 1.5e-8.
pub const ACCURACY: f64 = 1e-10;

#[macro_export]
macro_rules! make_test_harness {
	($dist:ident($($arg_name:ident: $arg_type:ty),+), $err:ty) => {
		#[allow(unused_parens)]
		fn new_dist(($($arg_name),+): ($($arg_type),+)) -> $dist {
			match <$dist>::new($($arg_name),+) {
				Ok(dist) => dist,
				Err(err) => panic!(
					"Expected {}::new{:?} to return Ok(_), got {:?} instead",
					stringify!($dist),
					// args tuple
					($($arg_name),+),
					err,
				)
			}
		}

		#[allow(unused_parens)]
		fn assert_new_is_ok(args: ($($arg_type),+)) {
			new_dist(args);
		}

		#[allow(unused_parens)]
		fn assert_new_is_err(($($arg_name),+): ($($arg_type),+), err: $err) {
			match <$dist>::new($($arg_name),+) {
				Err(this_err) if this_err == err => return,
				result @ Ok(_) | result @ Err(_) => panic!(
					"Expected {}::new{:?} to return {:?}, got {:?} instead",
					stringify!($dist),
					($($arg_name),+),
					err,
					result,
				)
			}

		}

		$crate::make_test_harness!(@common, $dist($($arg_name: $arg_type),+));
	};

	($dist:ident($($arg_name:ident: $arg_type:ty),+)) => {
		#[allow(unused_parens)]
		fn new_dist(($($arg_name),+): ($($arg_type),+)) -> $dist {
			<$dist>::new($($arg_name),+)
		}

		$crate::make_test_harness!(@common, $dist($($arg_name: $arg_type),+));
	};

	(@common, $dist:ident($($arg_name:ident: $arg_type:ty),+)) => {
		#[allow(unused)]
		fn assert_exact<T, F, O>(
			new_args: ($($arg_type),+),
			op_args: T,
			op: F,
			expected: O,
		) where
			T: std::fmt::Debug + Copy,
			F: Fn($dist, T) -> O,
			O: std::fmt::Debug + PartialEq,
		{
			let dist = new_dist(new_args);
			let value = op(dist, op_args);

			if expected == value {
				return;
			}

			panic!(
				"Expected {}{:?} with {:?} to return {:?}, got {:?} instead",
				stringify!($dist),
				new_args,
				op_args,
				expected,
				value,
			)

		}

		#[allow(unused_parens)]
		#[allow(unused)]
		fn assert_close<T, F>(
			new_args: ($($arg_type),+),
			op_args: T,
			op: F,
			expected: f64
		) where
			T: std::fmt::Debug + Copy,
			F: Fn($dist, T) -> f64,
		{
			let dist = new_dist(new_args);
			let value = op(dist, op_args);


			if almost_eq!(expected, value, relative = ACCURACY) {
				return;
			}

			panic!(
				"Expected {}{:?} with {:?} to be close to {}, got {} instead",
				stringify!($dist),
				new_args,
				op_args,
				expected,
				value,
			)
		}
	}
}

mod test_make_test_harness {
	use stats::distribution::{Beta, BetaError};
	use stats::statistics::*;

	use crate::prelude::*;

	make_test_harness!(Beta(shape_a: f64, shape_b: f64), BetaError);

	#[test]
	fn test_new_is_ok_success() {
		assert_new_is_ok((0.8, 1.2));
	}

	#[test]
	#[should_panic]
	fn test_new_is_ok_failure() {
		assert_new_is_ok((-1.0, 1.0));
	}

	#[test]
	fn test_new_is_err_success() {
		assert_new_is_err((-0.8, 1.2), BetaError::InvalidAlpha);
	}

	#[test]
	#[should_panic]
	fn test_new_is_err_failure() {
		assert_new_is_err((0.8, 1.2), BetaError::InvalidAlpha);
	}

	#[test]
	fn test_exact_success() {
		assert_exact((1.5, 1.5), (), |d, _| d.mode(), Some(0.5));
	}

	#[test]
	#[should_panic]
	fn test_exact_failure() {
		assert_exact(
			(0.8, 1.2),
			(),
			|d, _| d.mode(),
			Some(0.333333333333),
		);
	}

	#[test]
	fn test_close_success() {
		assert_close(
			(1.2, 1.4),
			(),
			|d, _| d.mode().unwrap(),
			0.333333333333,
		);
	}

	#[test]
	#[should_panic]
	fn test_close_failure() {
		assert_close(
			(1.2, 1.4),
			(),
			|d, _| d.mode().unwrap(),
			0.3333333333,
		);
	}
}

/// cdf should be the integral of the pdf
fn check_integrate_pdf_is_cdf<D>(dist: &D, x_min: f64, x_max: f64, step: f64)
where
	D: ContinuousCDF + Continuous,
{
	let mut prev_x = x_min;
	let mut prev_density = dist.pdf(x_min);
	let mut sum = 0.0;

	loop {
		let x = prev_x + step;
		let density = dist.pdf(x);

		assert!(density >= 0.0);

		let ln_density = dist.ln_pdf(x);

		assert_almost_eq!(density.ln(), ln_density, epsilon = 1e-10);

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

		assert_almost_eq!(sum, dist.cdf(i), epsilon = 1e-10);
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
	D: ContinuousCDF + Continuous,
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

		assert_almost_eq!(d_cdf, DX * density, epsilon = 1e-11);

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
	D: ContinuousCDF + Continuous + std::panic::RefUnwindSafe,
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
		panic!(
			"Integration of pdf doesn't equal cdf and derivative of cdf doesn't equal pdf!"
		);
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

pub mod prelude {
	pub use math::{Probability, almost_eq, assert_almost_eq};
	pub use num_traits::Float;
	pub use stats::{
		distribution::{
			Continuous, ContinuousCDF, Discrete, DiscreteCDF,
		},
		statistics::{Distribution, Mode},
	};

	pub use super::{
		ACCURACY, check_continuous_distribution,
		check_discrete_distribution, make_test_harness,
	};
}
