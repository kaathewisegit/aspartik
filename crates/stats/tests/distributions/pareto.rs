use stats::distribution::{Pareto, ParetoError};

use crate::prelude::*;

make_test_harness!(Pareto(scale: f64, shape: f64), ParetoError);

#[test]
fn test_new_is_ok() {
	let cases = [
		(10.0, 0.1),
		(5.0, 1.0),
		(0.1, 10.0),
		(10.0, 100.0),
		(1.0, f64::INFINITY),
		(f64::INFINITY, f64::INFINITY),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((1.0, -1.0), ParetoError::ShapeInvalid),
		((-1.0, 1.0), ParetoError::ScaleInvalid),
		((0.0, 0.0), ParetoError::ScaleInvalid),
		((-1.0, -1.0), ParetoError::ScaleInvalid),
		((f64::NAN, 1.0), ParetoError::ScaleInvalid), // Note: ParetoError::LocationInvalid in Levy, but ParetoError::ScaleInvalid here based on original
		((1.0, f64::NAN), ParetoError::ShapeInvalid),
		((f64::NAN, f64::NAN), ParetoError::ScaleInvalid), // Note: ParetoError::LocationInvalid in Levy, but ParetoError::ScaleInvalid here based on original
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_variance() {
	let cases = [((1.0, 3.0), 0.75), ((10.0, 10.0), 125.0 / 81.0)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_variance_degen() {
	let cases = [((1.0, 1.0), ())]; // shape <= 2.0
	for (args, _) in cases {
		assert!(new_dist(args).variance().is_none());
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((0.1, 0.1), -11.0),
		((1.0, 1.0), -2.0),
		((10.0, 10.0), -1.1),
		((3.0, 1.0), -2.0 - 3f64.ln()),
		((1.0, 3.0), -4.0 / 3.0 + 3f64.ln()),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((1.0, 4.0), 5.0 * 2f64.sqrt()),
		((1.0, 100.0), (707.0 / 485.0) * 2f64.sqrt()),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_skewness_invalid_shape() {
	let cases = [((1.0, 3.0), ())];
	for (args, _) in cases {
		assert!(new_dist(args).skewness().is_none());
	}
}

#[test]
fn test_mode() {
	let cases = [
		((0.1, 1.0), 0.1),
		((2.0, 1.0), 2.0),
		((10.0, f64::INFINITY), 10.0),
		((f64::INFINITY, 1.0), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((0.1, 0.1), 102.4),
		((1.0, 1.0), 2.0),
		((10.0, 10.0), 10.0 * 2f64.powf(0.1)),
		((3.0, 0.5), 12.0),
		((10.0, f64::INFINITY), 10.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [
		((0.2, f64::INFINITY), 0.2),
		((10.0, f64::INFINITY), 10.0),
		((f64::INFINITY, 1.0), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((1.0, 0.1), f64::INFINITY), ((3.0, 10.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((1.0, 1.0), 0.1, 0.0),
		((1.0, 1.0), 1.0, 1.0),
		((1.0, 1.0), 1.5, 4.0 / 9.0),
		((1.0, 1.0), 5.0, 1.0 / 25.0),
		((1.0, 1.0), 50.0, 1.0 / 2500.0),
		((1.0, 4.0), 1.0, 4.0),
		((1.0, 4.0), 1.5, 128.0 / 243.0),
		((1.0, 4.0), 50.0, 1.0 / 78125000.0),
		((3.0, 2.0), 3.0, 2.0 / 3.0),
		((3.0, 2.0), 5.0, 18.0 / 125.0),
		((25.0, 100.0), 50.0, 1.5777218104420236e-30),
		((100.0, 25.0), 150.0, 6.6003546737276816e-6),
		((1.0, 2.0), f64::INFINITY, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.pdf(p_val), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((1.0, 1.0), 0.1, f64::NEG_INFINITY),
		((1.0, 1.0), 1.0, 0.0),
		((1.0, 1.0), 1.5, 4f64.ln() - 9f64.ln()),
		((1.0, 1.0), 5.0, -(25f64.ln())),
		((1.0, 1.0), 50.0, -(2500f64.ln())),
		((1.0, 4.0), 1.0, 4f64.ln()),
		((1.0, 4.0), 1.5, 128f64.ln() - 243f64.ln()),
		((1.0, 4.0), 50.0, -(78125000f64.ln())),
		((3.0, 2.0), 3.0, 2f64.ln() - 3f64.ln()),
		((3.0, 2.0), 5.0, 18f64.ln() - 125f64.ln()),
		((25.0, 100.0), 50.0, 1.5777218104420236e-30f64.ln()),
		((100.0, 25.0), 150.0, 6.6003546737276816e-6f64.ln()),
		((1.0, 2.0), f64::INFINITY, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.ln_pdf(p_val), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.1, 0.1), 0.1, 0.0),
		((1.0, 1.0), 1.0, 0.0),
		((5.0, 5.0), 2.0, 0.0),
		((7.0, 7.0), 10.0, 0.9176457),
		((10.0, 10.0), 12.0, 50700551.0 / 60466176.0),
		((5.0, 1.0), 10.0, 0.5),
		((3.0, 10.0), 6.0, 1023.0 / 1024.0),
		((1.0, 1.0), f64::INFINITY, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.cdf(p_val), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.1, 0.1), 0.1, 1.0),
		((1.0, 1.0), 1.0, 1.0),
		((5.0, 5.0), 2.0, 1.0),
		((7.0, 7.0), 10.0, 0.08235429999999999),
		((10.0, 10.0), 12.0, 0.16150558288984573),
		((5.0, 1.0), 10.0, 0.5),
		((3.0, 10.0), 6.0, 0.0009765625),
		((1.0, 1.0), f64::INFINITY, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.sf(p_val), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0.1, 0.1), 0.1, 0.1),
		((1.0, 1.0), 1.0, 1.0),
		((7.0, 7.0), 10.0, 10.0),
		((10.0, 10.0), 12.0, 12.0),
		((5.0, 1.0), 10.0, 10.0),
		((3.0, 10.0), 6.0, 6.0),
	];
	for (args, p, expected) in cases {
		// The original test uses `x.inverse_cdf(x.cdf(arg))`, which
		// implies we're testing if inverse_cdf correctly reverses cdf
		// for the given `arg`.  The new harness's `assert_exact` takes
		// `(args, input_val, closure, expected)`.  Here, the
		// `input_val` to `inverse_cdf` is `d.cdf(p)`, and the
		// `expected` is `p`.
		assert_exact(
			args,
			p,
			|d, p| {
				let prob = Probability::new(d.cdf(p));
				d.inverse_cdf(prob)
			},
			expected,
		);
	}
}

#[test]
fn test_continuous() {
	let cases = [((1.0, 10.0), 1.0, 10.0), ((0.1, 2.0), 0.1, 100.0)];
	for (args, lower, upper) in cases {
		check_continuous_distribution(&new_dist(args), lower, upper);
	}
}
