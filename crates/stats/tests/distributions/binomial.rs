use stats::distribution::{Binomial, BinomialError};

use crate::prelude::*;

make_test_harness!(Binomial(p: f64, n: u64), BinomialError);

#[test]
fn test_new_is_ok() {
	let cases = [(0.0, 4), (0.3, 3), (1.0, 2)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((f64::NAN, 1), BinomialError::ProbabilityInvalid),
		((-1.0, 1), BinomialError::ProbabilityInvalid),
		((2.0, 1), BinomialError::ProbabilityInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [((0.0, 4), 0.0), ((0.3, 3), 0.9), ((1.0, 2), 2.0)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [((0.0, 4), 0.0), ((0.3, 3), 0.63), ((1.0, 2), 0.0)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((0.0, 4), 0.0),
		((0.3, 3), 1.1404671643037713),
		((1.0, 2), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((0.0, 4), f64::INFINITY),
		((0.3, 3), 0.5039526306789697),
		((1.0, 2), f64::NEG_INFINITY),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [((0.0, 4), 0.0), ((0.3, 3), 0.0), ((1.0, 2), 2.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [((0.0, 4), 0), ((0.3, 3), 1), ((1.0, 2), 2)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [((0.3, 10), 0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((0.3, 10), 10)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pmf() {
	let cases = [
		((0.0, 1), 0, 1.0),
		((0.0, 1), 1, 0.0),
		((0.0, 3), 0, 1.0),
		((0.0, 3), 1, 0.0),
		((0.0, 3), 3, 0.0),
		((0.0, 10), 0, 1.0),
		((0.0, 10), 1, 0.0),
		((0.0, 10), 10, 0.0),
		((0.3, 1), 0, 0.7),
		((0.3, 1), 1, 0.3),
		((0.3, 3), 0, 0.343),
		((0.3, 3), 1, 0.441),
		((0.3, 3), 3, 0.027),
		((0.3, 10), 0, 0.0282475249),
		((0.3, 10), 1, 0.121060821),
		((0.3, 10), 10, 0.0000059049),
		((1.0, 1), 0, 0.0),
		((1.0, 1), 1, 1.0),
		((1.0, 3), 0, 0.0),
		((1.0, 3), 1, 0.0),
		((1.0, 3), 3, 1.0),
		((1.0, 10), 0, 0.0),
		((1.0, 10), 1, 0.0),
		((1.0, 10), 10, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.pmf(p_val), expected);
	}
}

#[test]
fn test_ln_pmf() {
	let cases = [
		((0.0, 1), 0, 0.0),
		((0.0, 1), 1, f64::NEG_INFINITY),
		((0.0, 3), 0, 0.0),
		((0.0, 3), 1, f64::NEG_INFINITY),
		((0.0, 3), 3, f64::NEG_INFINITY),
		((0.0, 10), 0, 0.0),
		((0.0, 10), 1, f64::NEG_INFINITY),
		((0.0, 10), 10, f64::NEG_INFINITY),
		((0.3, 1), 0, -0.35667494393873244),
		((0.3, 1), 1, -1.203972804325936),
		((0.3, 3), 0, -1.0700248318161973),
		((0.3, 3), 1, -0.8187104035352912),
		((0.3, 3), 3, -3.611918412977808),
		((0.3, 10), 0, -3.5667494393873244),
		((0.3, 10), 1, -2.1114622067804823),
		((0.3, 10), 10, -12.03972804325936),
		((1.0, 1), 0, f64::NEG_INFINITY),
		((1.0, 1), 1, 0.0),
		((1.0, 3), 0, f64::NEG_INFINITY),
		((1.0, 3), 1, f64::NEG_INFINITY),
		((1.0, 3), 3, 0.0),
		((1.0, 10), 0, f64::NEG_INFINITY),
		((1.0, 10), 1, f64::NEG_INFINITY),
		((1.0, 10), 10, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.ln_pmf(p_val), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.0, 1), 0, 1.0),
		((0.0, 1), 1, 1.0),
		((0.0, 3), 0, 1.0),
		((0.0, 3), 1, 1.0),
		((0.0, 3), 3, 1.0),
		((0.0, 10), 0, 1.0),
		((0.0, 10), 1, 1.0),
		((0.0, 10), 10, 1.0),
		((0.3, 1), 0, 0.7),
		((0.3, 1), 1, 1.0),
		((0.3, 3), 0, 0.343),
		((0.3, 3), 1, 0.784),
		((0.3, 3), 3, 1.0),
		((0.3, 10), 0, 0.0282475249),
		((0.3, 10), 1, 0.1493083459),
		((0.3, 10), 10, 1.0),
		((1.0, 1), 0, 0.0),
		((1.0, 1), 1, 1.0),
		((1.0, 3), 0, 0.0),
		((1.0, 3), 1, 0.0),
		((1.0, 3), 3, 1.0),
		((1.0, 10), 0, 0.0),
		((1.0, 10), 1, 0.0),
		((1.0, 10), 10, 1.0),
		// upper bound
		((0.5, 3), 5, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.cdf(p_val), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.0, 1), 0, 0.0),
		((0.0, 1), 1, 0.0),
		((0.0, 3), 0, 0.0),
		((0.0, 3), 1, 0.0),
		((0.0, 3), 3, 0.0),
		((0.0, 10), 0, 0.0),
		((0.0, 10), 1, 0.0),
		((0.0, 10), 10, 0.0),
		((0.3, 1), 0, 0.3),
		((0.3, 1), 1, 0.0),
		((0.3, 3), 0, 0.657),
		((0.3, 3), 1, 0.216),
		((0.3, 3), 3, 0.0),
		((0.3, 10), 0, 0.9717524751),
		((0.3, 10), 1, 0.8506916541),
		((0.3, 10), 10, 0.0),
		((1.0, 1), 0, 1.0),
		((1.0, 1), 1, 0.0),
		((1.0, 3), 0, 1.0),
		((1.0, 3), 1, 1.0),
		((1.0, 3), 3, 0.0),
		((1.0, 10), 0, 1.0),
		((1.0, 10), 1, 1.0),
		((1.0, 10), 10, 0.0),
		// upper bound
		((0.5, 3), 5, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.sf(p_val), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0.4, 5), 0.3456, 2),
		((0.018, 465), 3.472e-4, 1), // issue #185
		((0.5, 6), 0.75, 4),         // issue #185
		((0.05, 2), 0.5, 0),         // issue #330
		((0.005, 10), 0.9, 0),       // issue #330
	];
	for (args, p, expected) in cases {
		let p = Probability::new(p);
		assert_exact(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}

#[test]
fn test_cdf_inverse_identity() {
	let cases = [((0.3, 10), 3), ((0.3, 10), 4), ((0.5, 6), 4)];
	for (args, p) in cases {
		assert_exact(
			args,
			p,
			|d, p| d.inverse_cdf(Probability::new(d.cdf(p))),
			p,
		);
	}
}

#[test]
fn test_discrete() {
	check_discrete_distribution(&new_dist((0.3, 5)), 5);
	check_discrete_distribution(&new_dist((0.7, 10)), 10);
}
