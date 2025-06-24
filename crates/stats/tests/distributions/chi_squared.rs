use stats::distribution::{ChiSquared, GammaError};

use crate::prelude::*;

make_test_harness!(ChiSquared(freedom: f64), GammaError);

#[test]
fn test_new_is_ok() {
	let cases = [10e-10, 1.0, 10e10];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		(0.0, GammaError::ShapeInvalid),
		(-1.0, GammaError::ShapeInvalid),
		(f64::NAN, GammaError::ShapeInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_median() {
	let cases = [
		(0.5, 0.08573388203017833),
		(1.0, 1.0 - 2.0 / 3.0),
		(2.0, 2.0 - 2.0 / 3.0),
		(2.5, 2.5 - 2.0 / 3.0),
		(3.0, 3.0 - 2.0 / 3.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [
		(1.0, f64::INFINITY),
		(2.0, f64::INFINITY),
		(3.0, f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_continuous() {
	check_continuous_distribution(&new_dist(1.0), 0.0, 10.0);
	check_continuous_distribution(&new_dist(2.0), 0.0, 10.0);
	check_continuous_distribution(&new_dist(5.0), 0.0, 50.0);
}
