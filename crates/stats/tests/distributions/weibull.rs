use stats::distribution::{Weibull, WeibullError};

use std::f64::consts::{LN_2, LN_10};

use crate::prelude::*;

make_test_harness!(Weibull(shape: f64, scale: f64), WeibullError);

#[test]
fn test_new_is_ok() {
	let cases =
		[(1.0, 0.1), (10.0, 1.0), (11.0, 10.0), (12.0, f64::INFINITY)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((f64::NAN, 1.0), WeibullError::ShapeInvalid),
		((1.0, f64::NAN), WeibullError::ScaleInvalid),
		((f64::NAN, f64::NAN), WeibullError::ShapeInvalid), // Assuming ShapeInvalid takes precedence
		((1.0, -1.0), WeibullError::ScaleInvalid),
		((-1.0, 1.0), WeibullError::ShapeInvalid),
		((-1.0, -1.0), WeibullError::ShapeInvalid),
		((0.0, 0.0), WeibullError::ShapeInvalid),
		((0.0, 1.0), WeibullError::ShapeInvalid),
		((1.0, 0.0), WeibullError::ScaleInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((1.0, 0.1), 0.1),
		((1.0, 1.0), 1.0),
		((10.0, 10.0), 9.513507698668732),
		((10.0, 1.0), 0.9513507698668732),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((1.0, 0.1), 0.01),
		((1.0, 1.0), 1.0),
		((10.0, 10.0), 1.310045507346831),
		((10.0, 1.0), 0.01310045507346831),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((1.0, 0.1), -1.3025850929940457),
		((1.0, 1.0), 1.0),
		((10.0, 10.0), 1.5194940984113796),
		((10.0, 1.0), -0.7830909945826661),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((1.0, 0.1), 2.0),
		((1.0, 1.0), 2.0),
		((10.0, 10.0), -0.6376371339031444),
		((10.0, 1.0), -0.6376371339031444),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((1.0, 0.1), 0.06931471805599453),
		((1.0, 1.0), LN_2),
		((10.0, 10.0), 9.640122354677898),
		((10.0, 1.0), 0.9640122354677897),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((1.0, 0.1), 0.0),
		((1.0, 1.0), 0.0),
		((10.0, 10.0), 9.895192582062144),
		((10.0, 1.0), 0.9895192582062144),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [((1.0, 1.0), 0.0), ((10.0, 1.0), 0.0)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((1.0, 1.0), f64::INFINITY), ((10.0, 1.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((1.0, 0.1), 0.0, 10.0),
		((1.0, 0.1), 1.0, 0.0004539992976248485),
		((1.0, 0.1), 10.0, 3.720075976020836e-43),
		((1.0, 1.0), 0.0, 1.0),
		((1.0, 1.0), 1.0, 0.36787944117144233),
		((1.0, 1.0), 10.0, 0.000045399929762484854),
		((10.0, 10.0), 0.0, 0.0),
		((10.0, 10.0), 1.0, 9.999999999e-10),
		((10.0, 10.0), 10.0, 0.36787944117144233),
		((10.0, 1.0), 0.0, 0.0),
		((10.0, 1.0), 1.0, 3.6787944117144233),
		((10.0, 1.0), 10.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((1.0, 0.1), 0.0, LN_10),
		((1.0, 0.1), 1.0, -7.697414907005954),
		((1.0, 0.1), 10.0, -97.69741490700595),
		((1.0, 1.0), 0.0, 0.0),
		((1.0, 1.0), 1.0, -1.0),
		((1.0, 1.0), 10.0, -10.0),
		((10.0, 10.0), 0.0, f64::NEG_INFINITY),
		((10.0, 10.0), 1.0, -20.723265837046412),
		((10.0, 10.0), 10.0, -1.0),
		((10.0, 1.0), 0.0, f64::NEG_INFINITY),
		((10.0, 1.0), 1.0, 1.3025850929940457),
		((10.0, 1.0), 10.0, -9.99999997697415e9),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((1.0, 0.1), 0.0, 0.0),
		((1.0, 0.1), 1.0, 0.9999546000702375),
		(
			(1.0, 0.1),
			10.0,
			0.99999999999999999999999999999999999999999996279924,
		),
		((1.0, 1.0), 0.0, 0.0),
		((1.0, 1.0), 1.0, 0.6321205588285577),
		((1.0, 1.0), 10.0, 0.9999546000702375),
		((10.0, 10.0), 0.0, 0.0),
		((10.0, 10.0), 1.0, 9.9999999995e-11),
		((10.0, 10.0), 10.0, 0.6321205588285577),
		((10.0, 1.0), 0.0, 0.0),
		((10.0, 1.0), 1.0, 0.6321205588285577),
		((10.0, 1.0), 10.0, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((1.0, 0.1), 0.0, 1.0),
		((1.0, 0.1), 1.0, 4.5399929762484854e-5),
		((1.0, 0.1), 10.0, 3.720075976020836e-44),
		((1.0, 1.0), 0.0, 1.0),
		((1.0, 1.0), 1.0, 0.36787944117144233),
		((1.0, 1.0), 10.0, 4.5399929762484854e-5),
		((10.0, 10.0), 0.0, 1.0),
		((10.0, 10.0), 1.0, 0.9999999999),
		((10.0, 10.0), 10.0, 0.36787944117144233),
		((10.0, 1.0), 0.0, 1.0),
		((10.0, 1.0), 1.0, 0.36787944117144233),
		((10.0, 1.0), 10.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((1.0, 0.1), 0.0, 0.0), // cdf(0.0) for this distribution is 0.0
		((1.0, 0.1), 0.9999546000702375, 1.0), // using cdf(1.0)
		((1.0, 1.0), 0.0, 0.0), // cdf(0.0)
		((1.0, 1.0), 0.6321205588285577, 1.0), // cdf(1.0)
		((1.0, 1.0), 0.9999546000702375, 10.0), // cdf(10.0)
		((10.0, 10.0), 0.0, 0.0), // cdf(0.0)
		((10.0, 10.0), 9.9999999995e-11, 1.0), // cdf(1.0)
		((10.0, 10.0), 0.6321205588285577, 10.0), // cdf(10.0)
		((10.0, 1.0), 0.0, 0.0), // cdf(0.0)
		((10.0, 1.0), 0.6321205588285577, 1.0), // cdf(1.0)
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		assert_almost_eq!(
			dist.inverse_cdf(p),
			expected,
			relative = ACCURACY
		);
	}
}

#[test]
fn test_continuous() {
	let cases = [((1.0, 0.2), 0.0, 10.0)];
	for (args, lower, upper) in cases {
		check_continuous_distribution(&new_dist(args), lower, upper);
	}
}
