use stats::distribution::{Beta, BetaError};

use crate::prelude::*;

make_test_harness!(Beta(a: f64, b: f64), BetaError);

#[test]
fn test_new_is_ok() {
	let cases = [(1.0, 1.0), (9.0, 1.0), (5.0, 100.0)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((0.0, 0.0), BetaError::InvalidAlpha),
		((0.0, 0.1), BetaError::InvalidAlpha),
		((1.0, 0.0), BetaError::InvalidBeta),
		((0.5, f64::INFINITY), BetaError::InvalidBeta),
		((f64::INFINITY, 0.5), BetaError::InvalidAlpha),
		((f64::NAN, 1.0), BetaError::InvalidAlpha),
		((1.0, f64::NAN), BetaError::InvalidBeta),
		((f64::NAN, f64::NAN), BetaError::InvalidAlpha),
		((1.0, -1.0), BetaError::InvalidBeta),
		((-1.0, 1.0), BetaError::InvalidAlpha),
		((-1.0, -1.0), BetaError::InvalidAlpha),
		((f64::INFINITY, f64::INFINITY), BetaError::InvalidAlpha),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((1.0, 1.0), 0.5),
		((9.0, 1.0), 0.9),
		((5.0, 100.0), 0.047619047619047616),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((1.0, 1.0), 1.0 / 12.0),
		((9.0, 1.0), 9.0 / 1100.0),
		((5.0, 100.0), 500.0 / 1168650.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((9.0, 1.0), -1.3083356884473305),
		((5.0, 100.0), -2.520162318760274),
		((1.0, 1.0), 0.0),
	];
	for (args, expected) in cases {
		assert_almost_eq!(
			new_dist(args).entropy().unwrap(),
			expected,
			epsilon = f64::EPSILON * 4.0,
			relative = 1e-14,
		);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((1.0, 1.0), 0.0),
		((9.0, 1.0), -1.4740554623801778),
		((5.0, 100.0), 0.8175941092755343),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [((5.0, 100.0), 0.038834951456310676)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_mode_none() {
	let cases = [((1.0, 5.0), None), ((5.0, 1.0), None)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [((1.0, 1.0), 0.0), ((1e10, 1e-10), 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((1.0, 1.0), 1.0), ((1e10, 1e-10), 1.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((1.0, 1.0), 0.0, 1.0),
		((1.0, 1.0), 0.5, 1.0),
		((1.0, 1.0), 1.0, 1.0),
		((9.0, 1.0), 0.0, 0.0),
		((9.0, 1.0), 0.5, 0.03515625),
		((9.0, 1.0), 1.0, 9.0),
		((5.0, 100.0), 0.0, 0.0),
		((5.0, 100.0), 0.5, 4.5341022983503377e-23),
		((5.0, 100.0), 1.0, 0.0),
		((5.0, 100.0), 1.0, 0.0),
		// lt_0
		((1.0, 1.0), -1.0, 0.0),
		// gt_1
		((1.0, 1.0), 2.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((1.0, 1.0), 0.0, 0.0),
		((1.0, 1.0), 0.5, 0.0),
		((1.0, 1.0), 1.0, 0.0),
		((9.0, 1.0), 0.0, f64::NEG_INFINITY),
		((9.0, 1.0), 0.5, -3.347952867143343),
		((9.0, 1.0), 1.0, 2.1972245773362196),
		((5.0, 100.0), 0.0, f64::NEG_INFINITY),
		((5.0, 100.0), 0.5, -51.44783002453768),
		((5.0, 100.0), 1.0, f64::NEG_INFINITY),
		// lt_0
		((1.0, 1.0), -1.0, f64::NEG_INFINITY),
		// gt_1
		((1.0, 1.0), 2.0, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((1.0, 1.0), 0.0, 0.0),
		((1.0, 1.0), 0.5, 0.5),
		((1.0, 1.0), 1.0, 1.0),
		((9.0, 1.0), 0.0, 0.0),
		((9.0, 1.0), 0.5, 0.001953125),
		((9.0, 1.0), 1.0, 1.0),
		((5.0, 100.0), 0.0, 0.0),
		((5.0, 100.0), 0.5, 1.0),
		((5.0, 100.0), 1.0, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((1.0, 1.0), 0.0, 1.0),
		((1.0, 1.0), 0.5, 0.5),
		((1.0, 1.0), 1.0, 0.0),
		((9.0, 1.0), 0.0, 1.0),
		((9.0, 1.0), 0.5, 0.998046875),
		((9.0, 1.0), 1.0, 0.0),
		((5.0, 100.0), 0.0, 1.0),
		((5.0, 100.0), 0.5, 0.0),
		((5.0, 100.0), 1.0, 0.0),
		// lt_0
		((1.0, 1.0), -1.0, 1.0),
		// gt_1
		((1.0, 1.0), 2.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((1.0, 1.0), 0.0, 0.0),
		((1.0, 1.0), 0.5, 0.5),
		((1.0, 1.0), 1.0, 1.0),
		((9.0, 1.0), 0.0, 0.0),
		((9.0, 1.0), 0.001953125, 0.001953125),
		((9.0, 1.0), 0.5, 0.5),
		((9.0, 1.0), 1.0, 1.0),
		((5.0, 100.0), 0.0, 0.0),
		((5.0, 100.0), 0.01, 0.01),
		((5.0, 100.0), 1.0, 1.0),
		// lt_0
		((1.0, 1.0), -1.0, 0.0),
		// gt_1
		((1.0, 1.0), 2.0, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.inverse_cdf(d.cdf(p)), expected);
	}
}

#[test]
fn test_continuous() {
	check_continuous_distribution(&new_dist((1.2, 3.4)), 0.0, 1.0);
	check_continuous_distribution(&new_dist((4.5, 6.7)), 0.0, 1.0);
}
