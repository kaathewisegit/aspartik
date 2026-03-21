use stats::distribution::{Triangular, TriangularError};

use crate::prelude::*;

make_test_harness!(Triangular(min: f64, max: f64, mode: f64), TriangularError);

#[test]
fn test_new_is_ok() {
	let cases = [
		(-1.0, 1.0, 0.0),
		(1.0, 2.0, 1.0),
		(5.0, 25.0, 25.0),
		(1.0e-5, 1.0e5, 1.0e-3),
		(0.0, 1.0, 0.9),
		(-4.0, -0.5, -2.0),
		(-13.039, 8.42, 1.17),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((0.0, 0.0, 0.0), TriangularError::MinEqualsMax),
		((0.0, 1.0, -0.1), TriangularError::ModeOutOfRange),
		((0.0, 1.0, 1.1), TriangularError::ModeOutOfRange),
		((0.0, -1.0, 0.5), TriangularError::ModeOutOfRange),
		((2.0, 1.0, 1.5), TriangularError::ModeOutOfRange),
		((f64::NAN, 1.0, 0.5), TriangularError::MinInvalid),
		((0.2, f64::NAN, 0.5), TriangularError::MaxInvalid),
		((0.5, 1.0, f64::NAN), TriangularError::ModeInvalid),
		((f64::NAN, f64::NAN, f64::NAN), TriangularError::MinInvalid),
		((f64::NEG_INFINITY, 1.0, 0.5), TriangularError::MinInvalid),
		((0.0, f64::INFINITY, 0.5), TriangularError::MaxInvalid),
	];

	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((0.0, 1.0, 0.5), 0.75 / 18.0),
		((0.0, 1.0, 0.75), 0.8125 / 18.0),
		((-5.0, 8.0, -3.5), 151.75 / 18.0),
		((-5.0, 8.0, 5.0), 139.0 / 18.0),
		((-5.0, -3.0, -4.0), 3.0 / 18.0),
		((15.0, 134.0, 21.0), 13483.0 / 18.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((0.0, 1.0, 0.5), -0.19314718055994531),
		((0.0, 1.0, 0.75), -0.19314718055994531),
		((-5.0, 8.0, -3.5), 2.3718021769015913),
		((-5.0, 8.0, 5.0), 2.3718021769015913),
		((-5.0, -3.0, -4.0), 0.5),
		((15.0, 134.0, 21.0), 4.585976312551584),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((0.0, 1.0, 0.5), 0.0),
		((0.0, 1.0, 0.75), -0.4224039833745502),
		((-5.0, 8.0, -3.5), 0.5375093589712976),
		((-5.0, 8.0, 5.0), -0.44459917430125956),
		((-5.0, -3.0, -4.0), 0.0),
		((15.0, 134.0, 21.0), 0.560592092275186),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((0.0, 1.0, 0.5), 0.5),
		((0.0, 1.0, 0.75), 0.75),
		((-5.0, 8.0, -3.5), -3.5),
		((-5.0, 8.0, 5.0), 5.0),
		((-5.0, -3.0, -4.0), -4.0),
		((15.0, 134.0, 21.0), 21.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((0.0, 1.0, 0.5), 0.5),
		((0.0, 1.0, 0.75), 0.6123724356957945),
		((-5.0, 8.0, -3.5), -0.6458082328952913),
		((-5.0, 8.0, 5.0), 3.0622577482985496),
		((-5.0, -3.0, -4.0), -4.0),
		((15.0, 134.0, 21.0), 52.00304883716712),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((0.0, 1.0, 0.5), -1.0, 0.0),
		((0.0, 1.0, 0.5), 1.1, 0.0),
		((0.0, 1.0, 0.5), 0.25, 1.0),
		((0.0, 1.0, 0.5), 0.5, 2.0),
		((0.0, 1.0, 0.5), 0.75, 1.0),
		((-5.0, 8.0, -3.5), -5.1, 0.0),
		((-5.0, 8.0, -3.5), 8.1, 0.0),
		((-5.0, 8.0, -3.5), -4.0, 0.10256410256410256),
		((-5.0, 8.0, -3.5), -3.5, 0.15384615384615385),
		((-5.0, 8.0, -3.5), 4.0, 0.05351170568561873),
		((-5.0, -3.0, -4.0), -5.1, 0.0),
		((-5.0, -3.0, -4.0), -2.9, 0.0),
		((-5.0, -3.0, -4.0), -4.5, 0.5),
		((-5.0, -3.0, -4.0), -4.0, 1.0),
		((-5.0, -3.0, -4.0), -3.5, 0.5),
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p_val| d.pdf(p_val), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((0.0, 1.0, 0.5), -1.0, f64::NEG_INFINITY),
		((0.0, 1.0, 0.5), 1.1, f64::NEG_INFINITY),
		((0.0, 1.0, 0.5), 0.25, 1.0f64.ln()),
		((0.0, 1.0, 0.5), 0.5, 2f64.ln()),
		((0.0, 1.0, 0.5), 0.75, 1.0f64.ln()),
		((-5.0, 8.0, -3.5), -5.1, f64::NEG_INFINITY),
		((-5.0, 8.0, -3.5), 8.1, f64::NEG_INFINITY),
		((-5.0, 8.0, -3.5), -4.0, 0.10256410256410256_f64.ln()),
		((-5.0, 8.0, -3.5), -3.5, 0.15384615384615385_f64.ln()),
		((-5.0, 8.0, -3.5), 4.0, 0.05351170568561873_f64.ln()),
		((-5.0, -3.0, -4.0), -5.1, f64::NEG_INFINITY),
		((-5.0, -3.0, -4.0), -2.9, f64::NEG_INFINITY),
		((-5.0, -3.0, -4.0), -4.5, 0.5f64.ln()),
		((-5.0, -3.0, -4.0), -4.0, 0.0),
		((-5.0, -3.0, -4.0), -3.5, 0.5f64.ln()),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.ln_pdf(p_val), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.0, 1.0, 0.5), 0.25, 0.125),
		((0.0, 1.0, 0.5), 0.5, 0.5),
		((0.0, 1.0, 0.5), 0.75, 0.875),
		((-5.0, 8.0, -3.5), -4.0, 0.05128205128205128),
		((-5.0, 8.0, -3.5), -3.5, 0.11538461538461539),
		((-5.0, 8.0, -3.5), 4.0, 0.8929765886287625),
		((-5.0, -3.0, -4.0), -4.5, 0.125),
		((-5.0, -3.0, -4.0), -4.0, 0.5),
		((-5.0, -3.0, -4.0), -3.5, 0.875),
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p_val| d.cdf(p_val), expected);
	}
}

#[test]
fn test_cdf_lower_bound() {
	let cases = [((0.0, 3.0, 1.5), -1.0, 0.0)];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p_val| d.cdf(p_val), expected);
	}
}

#[test]
fn test_cdf_upper_bound() {
	let cases = [((0.0, 3.0, 1.5), 5.0, 1.0)];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p_val| d.cdf(p_val), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.0, 1.0, 0.5), 0.25, 0.875),
		((0.0, 1.0, 0.5), 0.5, 0.5),
		((0.0, 1.0, 0.5), 0.75, 0.125),
		((-5.0, 8.0, -3.5), -4.0, 0.9487179487179487),
		((-5.0, 8.0, -3.5), -3.5, 0.8846153846153846),
		((-5.0, 8.0, -3.5), 4.0, 0.10702341137123746),
		((-5.0, -3.0, -4.0), -4.5, 0.875),
		((-5.0, -3.0, -4.0), -4.0, 0.5),
		((-5.0, -3.0, -4.0), -3.5, 0.125),
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p_val| d.sf(p_val), expected);
	}
}

#[test]
fn test_sf_lower_bound() {
	let cases = [((0.0, 3.0, 1.5), -1.0, 1.0)];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p_val| d.sf(p_val), expected);
	}
}

#[test]
fn test_sf_upper_bound() {
	let cases = [((0.0, 3.0, 1.5), 5.0, 0.0)];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p_val| d.sf(p_val), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0.0, 1.0, 0.5), 0.125, 0.25),
		((0.0, 1.0, 0.5), 0.5, 0.5),
		((0.0, 1.0, 0.5), 0.875, 0.75),
		((-5.0, 8.0, -3.5), 0.05128205128205128, -4.0),
		((-5.0, 8.0, -3.5), 0.11538461538461539, -3.5),
		((-5.0, 8.0, -3.5), 0.8929765886287625, 4.0),
		((-5.0, -3.0, -4.0), 0.125, -4.5),
		((-5.0, -3.0, -4.0), 0.5, -4.0),
		((-5.0, -3.0, -4.0), 0.875, -3.5),
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		let p = Probability::new(p);
		assert_almost_eq!(dist.inverse_cdf(p), expected);
	}
}

#[test]
fn test_continuous() {
	let cases = [
		((-5.0, 5.0, 0.0), -5.0, 5.0),
		((-15.0, -2.0, -3.0), -15.0, -2.0),
	];
	for (args, lower, upper) in cases {
		check_continuous_distribution(&new_dist(args), lower, upper);
	}
}
