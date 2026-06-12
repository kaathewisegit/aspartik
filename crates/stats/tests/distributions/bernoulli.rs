use stats::distribution::Bernoulli;

use crate::prelude::*;

make_test_harness!(Bernoulli(p: f64));

#[test]
fn test_cdf() {
	let cases = [
		(0.3, 1, 1.0),
		(0.0, 0, 1.0),
		(0.0, 1, 1.0),
		(0.3, 0, 0.7),
		(0.7, 0, 0.3),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		(0.3, 1, 0.0),
		(0.0, 0, 0.0),
		(0.0, 1, 0.0),
		(0.3, 0, 0.3),
		(0.7, 0, 0.7),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		(0.0, 0.0, 0),
		(0.0, 0.5, 0),
		(1.0, 0.0, 0),
		(1.0, 1.0, 1),
		(1.0, 1e-6, 1),
		(0.5, 0.25, 0),
		(0.5, 0.5, 0),
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}
