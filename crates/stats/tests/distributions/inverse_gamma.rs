use stats::distribution::{InverseGamma, InverseGammaError};

use crate::prelude::*;

make_test_harness!(InverseGamma(shape: f64, rate: f64), InverseGammaError);

#[test]
fn test_new_is_ok() {
	let cases = [(0.1, 0.1), (1.0, 1.0)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((0.0, 1.0), InverseGammaError::ShapeInvalid),
		((1.0, -1.0), InverseGammaError::RateInvalid),
		((-1.0, 1.0), InverseGammaError::ShapeInvalid),
		((-100.0, 1.0), InverseGammaError::ShapeInvalid),
		((f64::NEG_INFINITY, 1.0), InverseGammaError::ShapeInvalid),
		((f64::NAN, 1.0), InverseGammaError::ShapeInvalid),
		((1.0, 0.0), InverseGammaError::RateInvalid),
		((1.0, -100.0), InverseGammaError::RateInvalid),
		((1.0, f64::NEG_INFINITY), InverseGammaError::RateInvalid),
		((1.0, f64::NAN), InverseGammaError::RateInvalid),
		((f64::INFINITY, 1.0), InverseGammaError::ShapeInvalid),
		((1.0, f64::INFINITY), InverseGammaError::RateInvalid),
		(
			(f64::INFINITY, f64::INFINITY),
			InverseGammaError::ShapeInvalid,
		),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [((1.1, 0.1), 1.0), ((1.1, 1.0), 10.0)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_mean_is_none_for_small_shape() {
	let cases = [(0.1, 0.1)];
	for args in cases {
		assert!(new_dist(args).mean().is_none());
	}
}

#[test]
fn test_variance() {
	let cases = [
		((2.1, 0.1), 0.08264462809917356),
		((2.1, 1.0), 8.264462809917354),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_variance_is_none_for_small_shape() {
	let cases = [(0.1, 0.1)];
	for args in cases {
		assert!(new_dist(args).variance().is_none());
	}
}

#[test]
fn test_entropy() {
	let cases = [
		// TODO: low precision
		// ((0.1, 0.1), 11.516257993192344),
		((1.0, 1.0), 2.1544313298030655),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((3.1, 0.1), 41.95235392680606),
		((3.1, 1.0), 41.95235392680606),
		((5.0, 0.1), 3.4641016151377544),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_skewness_is_none_for_small_shape() {
	let cases = [(0.1, 0.1)];
	for args in cases {
		assert!(new_dist(args).skewness().is_none());
	}
}

#[test]
fn test_mode() {
	let cases = [((0.1, 0.1), 0.09090909090909091), ((1.0, 1.0), 0.5)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [((1.0, 1.0), 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((1.0, 1.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((0.1, 0.1), 1.2, 0.0628591853882328),
		((0.1, 1.0), 2.0, 0.0297426109178249),
		((1.0, 0.1), 1.5, 0.041578088223627456),
		((1.0, 1.0), 1.2, 0.30180431146324876),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((0.1, 0.1), 1.2, 0.0628591853882328_f64.ln()),
		((0.1, 1.0), 2.0, 0.0297426109178249_f64.ln()),
		((1.0, 0.1), 1.5, 0.041578088223627456_f64.ln()),
		((1.0, 1.0), 1.2, 0.30180431146324876_f64.ln()),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.1, 0.1), 1.2, 0.18621519619460541),
		((0.1, 1.0), 2.0, 0.05859755410986648),
		((1.0, 0.1), 1.5, 0.9355069850316178),
		((1.0, 1.0), 1.2, 0.4345982085070782),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.1, 0.1), 1.2, 0.8137848038053936),
		((0.1, 1.0), 2.0, 0.9414024458901327),
		((1.0, 0.1), 1.5, 0.0644930149683822),
		((1.0, 1.0), 1.2, 0.565401791492922),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
#[ignore = "TODO underflow error"]
fn test_continuous() {
	check_continuous_distribution(&new_dist((1.0, 0.5)), 0.0, 100.0);
	check_continuous_distribution(&new_dist((9.0, 2.0)), 0.0, 100.0);
}
