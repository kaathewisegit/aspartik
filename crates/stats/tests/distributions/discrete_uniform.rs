use stats::distribution::{DiscreteUniform, DiscreteUniformError};

use crate::prelude::*;

make_test_harness!(
    DiscreteUniform(min: i64, max: i64),
    DiscreteUniformError
);

#[test]
fn test_new_is_ok() {
	let cases = [(10, 20), (-10, 10), (0, 4), (20, 20)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((-1, -2), DiscreteUniformError::MinMaxInvalid),
		((6, 5), DiscreteUniformError::MinMaxInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((-10, 10), 0.0),
		((0, 4), 2.0),
		((10, 20), 15.0),
		((20, 20), 20.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((-10, 10), 36.666666666666664),
		((0, 4), 2.0),
		((10, 20), 10.0),
		((20, 20), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((-10, 10), 3.044522437723423),
		((0, 4), 1.6094379124341003),
		((10, 20), 2.3978952727983707),
		((20, 20), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((-10, 10), 0.0),
		((0, 4), 0.0),
		((10, 20), 0.0),
		((20, 20), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((-10, 10), 0.0),
		((0, 4), 2.0),
		((10, 20), 15.0),
		((20, 20), 20.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases =
		[((-10, 10), 0), ((0, 4), 2), ((10, 20), 15), ((20, 20), 20)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_pmf() {
	let cases = [
		((-10, 10), -5, 0.047619047619047616),
		((-10, 10), 1, 0.047619047619047616),
		((-10, 10), 10, 0.047619047619047616),
		((-10, -10), 0, 0.0),
		((-10, -10), -10, 1.0),
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.pmf(x), expected);
	}
}

#[test]
fn test_ln_pmf() {
	let cases = [
		((-10, 10), -5, -3.044522437723423),
		((-10, 10), 1, -3.044522437723423),
		((-10, 10), 10, -3.044522437723423),
		((-10, -10), 0, f64::NEG_INFINITY),
		((-10, -10), -10, 0.0),
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.ln_pmf(x), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((-10, 10), -5, 0.2857142857142857),
		((-10, 10), 1, 0.5714285714285714),
		((-10, 10), 10, 1.0),
		((-10, -10), -10, 1.0),
		((0, 3), -1, 0.0), // test_cdf_lower_bound
		((0, 3), 5, 1.0),  // test_cdf_upper_bound
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.cdf(x), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((-10, 10), -5, 0.7142857142857143),
		((-10, 10), 1, 0.42857142857142855),
		((-10, 10), 10, 0.0),
		((-10, -10), -10, 0.0),
		((0, 3), -1, 1.0), // test_sf_lower_bound
	];
	for (args, x, expected) in cases {
		assert_close(args, x, |d, x| d.sf(x), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0, 0), 0.5, 0),
		((0, 0), 1.0, 0),
		((0, 5), 0.5, 2),
		((3, 10), 0.005, 3),
		((3, 10), 0.9995, 10),
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}
