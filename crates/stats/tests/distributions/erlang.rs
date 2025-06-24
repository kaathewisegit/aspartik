use stats::distribution::{Erlang, GammaError};

use crate::prelude::*;

make_test_harness!(Erlang(shape: u64, rate: f64), GammaError);

#[test]
fn test_new_is_ok() {
	let cases = [
		(1, 0.1),
		(1, 1.0),
		(10, 10.0),
		(10, 1.0),
		(10, f64::INFINITY),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((0, 1.0), GammaError::ShapeInvalid),
		((1, 0.0), GammaError::RateInvalid),
		((1, f64::NAN), GammaError::RateInvalid),
		((1, -1.0), GammaError::RateInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((1, 0.1), 10.0),
		((1, 1.0), 1.0),
		((10, 10.0), 1.0),
		((10, 1.0), 10.0),
		((10, f64::INFINITY), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

// TODO: test other statistics and methods

#[test]
fn test_continuous() {
	check_continuous_distribution(&new_dist((1, 2.5)), 0.0, 20.0);
	check_continuous_distribution(&new_dist((2, 1.5)), 0.0, 20.0);
	check_continuous_distribution(&new_dist((3, 0.5)), 0.0, 20.0);
}
