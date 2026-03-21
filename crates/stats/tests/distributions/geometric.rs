use stats::distribution::{Geometric, GeometricError};

use crate::prelude::*;

make_test_harness!(Geometric(p: f64), GeometricError);

#[test]
fn test_new_is_ok() {
	let cases = [0.3, 1.0];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		(f64::NAN, GeometricError::ProbabilityInvalid),
		(0.0, GeometricError::ProbabilityInvalid),
		(-1.0, GeometricError::ProbabilityInvalid),
		(2.0, GeometricError::ProbabilityInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [(0.3, 1.0 / 0.3), (1.0, 1.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [(0.3, 0.7 / (0.3 * 0.3)), (1.0, 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [(0.3, 2.9376363307689735)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
	assert!(new_dist(1.0).entropy().unwrap().is_nan());
}

#[test]
fn test_skewness() {
	let cases = [(0.3, 2.0318886358684694)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
	assert_exact(1.0, (), |d, _| d.skewness().unwrap(), f64::INFINITY);
}

#[test]
fn test_median() {
	let cases = [
		(0.0001, 6932.0),
		(0.1, 7.0),
		(0.3, 2.0),
		(0.9, 1.0),
		(1.0, 0.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [(0.3, 1), (1.0, 1)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [(0.3, 1)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [(0.3, u64::MAX)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pmf() {
	let cases = [
		(0.3, 1, 0.3),
		(0.3, 2, 0.21),
		(1.0, 1, 1.0),
		(1.0, 2, 0.0),
		(0.5, 1, 0.5),
		(0.5, 2, 0.25),
		// at zero
		(0.3, 0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pmf(p), expected);
	}
}

#[test]
fn test_ln_pmf() {
	let cases = [
		(0.3, 1, -1.203972804325936),
		(0.3, 2, -1.5606477482646683),
		(1.0, 1, 0.0),
		(1.0, 2, f64::NEG_INFINITY),
		// at zero
		(0.3, 0, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pmf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		(1.0, 1, 1.0),
		(1.0, 2, 1.0),
		(0.5, 1, 0.5),
		(0.5, 2, 0.75),
		(1e-9, 5, 4.99999999e-09),
		(1e-17, 10, 1e-16),
		(1e-17, 100000000000000, 0.0009995001666250085),
		// at zero
		(0.3, 0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		(1.0, 1, 0.0),
		(1.0, 2, 0.0),
		(0.5, 1, 0.5),
		(0.5, 2, 0.25),
		(1e-9, 5, 0.999999995),
		(1e-17, 10, 0.9999999999999999),
		(1e-17, 100000000000000, 0.999000499833375),
		// at zero
		(0.3, 0, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_discrete() {
	check_discrete_distribution(&new_dist(0.3), 100);
	check_discrete_distribution(&new_dist(0.6), 100);
	check_discrete_distribution(&new_dist(1.0), 1);
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		(1.0, 0.0, 1),
		(1.0, 1.0, 1),
		(0.2, 0.2, 1),
		(0.004, 0.5, 173),
	];
	for (args, p, expected) in cases {
		let p = Probability::new(p);
		assert_exact(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}

#[test]
fn test_inverse_cdf_roundtrip() {
	for p in [0.5, 0.1, 0.004] {
		for x in 1..50u64 {
			let d = new_dist(p);
			let cdf = d.cdf(x);
			let prob = Probability::new(cdf);
			assert_eq!(x, d.inverse_cdf(prob));
		}
	}
}
