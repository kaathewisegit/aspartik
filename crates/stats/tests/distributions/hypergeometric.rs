use stats::distribution::{Hypergeometric, HypergeometricError};

use std::f64::consts::{LN_2, LN_10};

use crate::prelude::*;

make_test_harness!(
    Hypergeometric(population: u64, successes: u64, draws: u64),
    HypergeometricError
);

#[test]
fn test_new_is_ok() {
	let cases = [
		(0, 0, 0),
		(1, 1, 1),
		(2, 1, 1),
		(2, 2, 2),
		(10, 1, 1),
		(10, 5, 3),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((2, 3, 2), HypergeometricError::TooManySuccesses),
		((10, 5, 20), HypergeometricError::TooManyDraws),
		((0, 1, 1), HypergeometricError::TooManySuccesses),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((1, 1, 1), 1.0),
		((2, 1, 1), 0.5),
		((2, 2, 2), 2.0),
		((10, 1, 1), 0.1),
		((10, 5, 3), 15.0 / 10.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_mean_is_none() {
	let cases = [(0, 0, 0)];
	for args in cases {
		assert!(new_dist(args).mean().is_none());
	}
}

#[test]
fn test_variance() {
	let cases = [
		((2, 1, 1), 0.25),
		((2, 2, 2), 0.0),
		((10, 1, 1), 81.0 / 900.0),
		((10, 5, 3), 525.0 / 900.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_variance_is_none() {
	let cases = [(1, 1, 1)];
	for args in cases {
		assert!(new_dist(args).variance().is_none());
	}
}

#[test]
fn test_skewness() {
	let cases = [((10, 1, 1), 8.0 / 3.0), ((10, 5, 3), 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_skewness_is_none() {
	let cases = [(2, 2, 2)];
	for args in cases {
		assert!(new_dist(args).skewness().is_none());
	}
}

#[test]
fn test_mode() {
	let cases = [
		((0, 0, 0), 0),
		((1, 1, 1), 1),
		((2, 1, 1), 1),
		((2, 2, 2), 2),
		((10, 1, 1), 0),
		((10, 5, 3), 2),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [
		((0, 0, 0), 0),
		((1, 1, 1), 1),
		((2, 1, 1), 0),
		((2, 2, 2), 2),
		((10, 1, 1), 0),
		((10, 5, 3), 0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [
		((0, 0, 0), 0),
		((1, 1, 1), 1),
		((2, 1, 1), 1),
		((2, 2, 2), 2),
		((10, 1, 1), 1),
		((10, 5, 3), 3),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pmf() {
	let cases = [
		((0, 0, 0), 0, 1.0),
		((1, 1, 1), 1, 1.0),
		((2, 1, 1), 0, 0.5),
		((2, 1, 1), 1, 0.5),
		((2, 2, 2), 2, 1.0),
		((10, 1, 1), 0, 0.9),
		((10, 1, 1), 1, 0.1),
		((10, 5, 3), 1, 0.4166666666666667),
		((10, 5, 3), 3, 0.08333333333333333),
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.pmf(x), expected);
	}
}

#[test]
fn test_ln_pmf() {
	let cases = [
		((0, 0, 0), 0, 0.0),
		((1, 1, 1), 1, 0.0),
		((2, 1, 1), 0, -LN_2),
		((2, 1, 1), 1, -LN_2),
		((2, 2, 2), 2, 0.0),
		((10, 1, 1), 0, -0.1053605156578263),
		((10, 1, 1), 1, -LN_10),
		((10, 5, 3), 1, -0.8754687373539),
		((10, 5, 3), 3, -2.4849066497880004),
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.ln_pmf(x), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((2, 1, 1), 0, 0.5),
		((10, 1, 1), 0, 0.9),
		((10, 5, 3), 1, 0.5),
		((10, 5, 3), 2, 11.0 / 12.0),
		((10000, 2, 9800), 0, 199.0 / 499950.0),
		((10000, 2, 9800), 1, 19799.0 / 499950.0),
		((0, 0, 0), 0, 1.0), // cdf_arg_too_big
		((2, 2, 2), 0, 0.0), // cdf_arg_too_small
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.cdf(x), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((2, 1, 1), 0, 0.5),
		((10, 1, 1), 0, 0.1),
		((10, 5, 3), 1, 0.5),
		((10, 5, 3), 2, 1.0 / 12.0),
		((10000, 2, 9800), 0, 499751.0 / 499950.0),
		((10000, 2, 9800), 1, 480151.0 / 499950.0),
		((0, 0, 0), 0, 0.0), // sf_arg_too_big
		((2, 2, 2), 0, 1.0), // sf_arg_too_small
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.sf(x), expected);
	}
}

#[test]
fn test_discrete() {
	check_discrete_distribution(&new_dist((5, 4, 3)), 4);
	check_discrete_distribution(&new_dist((3, 2, 1)), 2);
}

#[test]
fn test_inverse_cdf() {
	let cases = [((10, 2, 5), 0.5, 1), ((100, 2, 5), 0.5, 0)];
	for (args, p, expected) in cases {
		let p = Probability::new(p);
		assert_exact(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}
