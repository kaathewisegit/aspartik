use stats::distribution::{Uniform, UniformError};

use math::assert_almost_eq;
use std::f64::consts::{LN_2, LN_10};

use crate::prelude::*;

make_test_harness!(Uniform(min: f64, max: f64), UniformError);

#[test]
fn test_new_is_ok() {
	let cases = [(0.0, 0.1), (0.0, 1.0), (-5.0, 11.0), (-5.0, 100.0)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((0.0, 0.0), UniformError::MaxNotGreaterThanMin),
		((f64::NAN, 1.0), UniformError::MinInvalid),
		((1.0, f64::NAN), UniformError::MaxInvalid),
		((f64::NAN, f64::NAN), UniformError::MinInvalid),
		((0.0, f64::INFINITY), UniformError::MaxInvalid),
		((1.0, 0.0), UniformError::MaxNotGreaterThanMin),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((-0.0, 2.0), 1.0 / 3.0),
		((0.0, 2.0), 1.0 / 3.0),
		((0.1, 4.0), 1.2675),
		((10.0, 11.0), 1.0 / 12.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((-0.0, 2.0), LN_2),
		((0.0, 2.0), LN_2),
		((0.1, 4.0), 1.3609765531356008),
		((1.0, 10.0), 2.1972245773362196),
		((10.0, 11.0), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((-0.0, 2.0), 0.0),
		((0.0, 2.0), 0.0),
		((0.1, 4.0), 0.0),
		((1.0, 10.0), 0.0),
		((10.0, 11.0), 0.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((-0.0, 2.0), 1.0),
		((0.0, 2.0), 1.0),
		((0.1, 4.0), 2.05),
		((1.0, 10.0), 5.5),
		((10.0, 11.0), 10.5),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((-0.0, 2.0), 1.0),
		((0.0, 2.0), 1.0),
		((0.1, 4.0), 2.05),
		((1.0, 10.0), 5.5),
		((10.0, 11.0), 10.5),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((0.0, 0.1), -5.0, 0.0),
		((0.0, 0.1), 0.05, 10.0),
		((0.0, 0.1), 5.0, 0.0),
		((0.0, 1.0), -5.0, 0.0),
		((0.0, 1.0), 0.5, 1.0),
		((0.0, 0.1), 5.0, 0.0),
		((0.0, 10.0), -5.0, 0.0),
		((0.0, 10.0), 1.0, 0.1),
		((0.0, 10.0), 5.0, 0.1),
		((0.0, 10.0), 11.0, 0.0),
		((-5.0, 100.0), -10.0, 0.0),
		((-5.0, 100.0), -5.0, 0.009523809523809525),
		((-5.0, 100.0), 0.0, 0.009523809523809525),
		((-5.0, 100.0), 101.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((0.0, 0.1), -5.0, f64::NEG_INFINITY),
		((0.0, 0.1), 0.05, LN_10),
		((0.0, 0.1), 5.0, f64::NEG_INFINITY),
		((0.0, 1.0), -5.0, f64::NEG_INFINITY),
		((0.0, 1.0), 0.5, 0.0),
		((0.0, 0.1), 5.0, f64::NEG_INFINITY),
		((0.0, 10.0), -5.0, f64::NEG_INFINITY),
		((0.0, 10.0), 1.0, -LN_10),
		((0.0, 10.0), 5.0, -LN_10),
		((0.0, 10.0), 11.0, f64::NEG_INFINITY),
		((-5.0, 100.0), -10.0, f64::NEG_INFINITY),
		((-5.0, 100.0), -5.0, -4.653960350157523),
		((-5.0, 100.0), 0.0, -4.653960350157523),
		((-5.0, 100.0), 101.0, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.0, 0.1), 0.05, 0.5),
		((0.0, 1.0), 0.5, 0.5),
		((0.0, 10.0), 1.0, 0.1),
		((0.0, 10.0), 5.0, 0.5),
		((-5.0, 100.0), -5.0, 0.0),
		((-5.0, 100.0), 0.0, 0.047619047619047616),
		((0.0, 3.0), -1.0, 0.0), // test_cdf_lower_bound
		((0.0, 3.0), 5.0, 1.0),  // test_cdf_upper_bound
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0.0, 0.1), 0.5, 0.05),
		((0.0, 10.0), 0.5, 5.0),
		((1.0, 10.0), 0.0, 1.0),
		((1.0, 10.0), 1.0 / 3.0, 4.0),
		((1.0, 10.0), 1.0, 10.0),
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		let p = Probability::new(p);
		assert_almost_eq!(dist.inverse_cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.0, 0.1), 0.05, 0.5),
		((0.0, 1.0), 0.5, 0.5),
		((0.0, 10.0), 1.0, 0.9),
		((0.0, 10.0), 5.0, 0.5),
		((-5.0, 100.0), -5.0, 1.0),
		((-5.0, 100.0), 0.0, 0.9523809523809523),
		((0.0, 3.0), -1.0, 1.0), // test_sf_lower_bound
		((0.0, 3.0), 5.0, 0.0),  // test_sf_upper_bound
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_continuous() {
	let cases = [((0.0, 10.0), 0.0, 10.0), ((-2.0, 15.0), -2.0, 15.0)];
	for (args, lower, upper) in cases {
		check_continuous_distribution(&new_dist(args), lower, upper);
	}
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
#[test]
fn test_samples_in_range() {
	use rand::{SeedableRng, distr::Distribution};

	let seed = [
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	];
	let mut r = rand_pcg::Pcg64::from_seed(seed);

	let min = -0.5;
	let max = 0.5;
	let num_trials = 10_000;
	let n = new_dist((min, max));

	assert!((0..num_trials)
		.map(|_| n.sample(&mut r))
		.all(|v| (min <= v) && (v < max)));
}

#[test]
fn test_default() {
	let n = Uniform::default();

	let n_mean = n.mean().unwrap();
	let n_std = n.std_dev().unwrap();

	// Check that the mean of the distribution is close to 1/2
	assert_almost_eq!(n_mean, 0.5);
	// Check that the standard deviation of the distribution is close to
	// 1/sqrt(12)
	assert_almost_eq!(n_std, 0.2886751345948129);
}
