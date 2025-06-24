use stats::distribution::{Exp, ExpError};

use std::f64::consts::{LN_10, LN_2};

use crate::prelude::*;

make_test_harness!(Exp(rate: f64), ExpError);

#[test]
fn test_new_is_ok() {
	let cases = [0.1, 1.0, 10.0];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		(f64::NAN, ExpError::RateInvalid),
		(0.0, ExpError::RateInvalid),
		(-1.0, ExpError::RateInvalid),
		(-10.0, ExpError::RateInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [(0.1, 10.0), (1.0, 1.0), (10.0, 0.1)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [(0.1, 100.0), (1.0, 1.0), (10.0, 0.01)];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		(0.1, 3.302585092994046),
		(1.0, 1.0),
		(10.0, -1.3025850929940457),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [(0.1, 2.0), (1.0, 2.0), (10.0, 2.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		(0.1, 6.931471805599453),
		(1.0, LN_2),
		(10.0, 0.06931471805599453),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [(0.1, 0.0), (1.0, 0.0), (10.0, 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [(0.1, 0.0), (1.0, 0.0), (10.0, 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [
		(0.1, f64::INFINITY),
		(1.0, f64::INFINITY),
		(10.0, f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		(0.1, 0.0, 0.1),
		(1.0, 0.0, 1.0),
		(10.0, 0.0, 10.0),
		(0.1, 0.1, 0.09900498337491681),
		(1.0, 0.1, 0.9048374180359596),
		(10.0, 0.1, 3.6787944117144233),
		(0.1, 1.0, 0.09048374180359596),
		(1.0, 1.0, 0.36787944117144233),
		(10.0, 1.0, 4.539992976248485e-4),
		(0.1, f64::INFINITY, 0.0),
		(1.0, f64::INFINITY, 0.0),
		(10.0, f64::INFINITY, 0.0),
		(0.1, -1.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.pdf(p_val), expected);
	}
}

#[test]
fn test_pdf_nan() {
	let cases = [
		(f64::INFINITY, 0.0),
		(f64::INFINITY, 0.1),
		(f64::INFINITY, 1.0),
		(f64::INFINITY, f64::INFINITY),
	];
	for (args, p) in cases {
		assert!(new_dist(args).pdf(p).is_nan());
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		(0.1, 0.0, -LN_10),
		(1.0, 0.0, 0.0),
		(10.0, 0.0, LN_10),
		(0.1, 0.1, -2.3125850929940457),
		(1.0, 0.1, -0.1),
		(10.0, 0.1, 1.3025850929940457),
		(0.1, 1.0, -2.4025850929940455),
		(1.0, 1.0, -1.0),
		(10.0, 1.0, -7.697414907005954),
		(0.1, f64::INFINITY, f64::NEG_INFINITY),
		(1.0, f64::NEG_INFINITY, f64::NEG_INFINITY),
		(10.0, f64::NEG_INFINITY, f64::NEG_INFINITY),
		(0.1, -1.0, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.ln_pdf(p_val), expected);
	}
}

#[test]
fn test_ln_pdf_nan() {
	let cases = [
		(f64::INFINITY, 0.0),
		(f64::INFINITY, 0.1),
		(f64::INFINITY, 1.0),
		(f64::INFINITY, f64::INFINITY),
	];
	for (args, p) in cases {
		assert!(new_dist(args).pdf(p).is_nan());
	}
}

#[test]
fn test_cdf() {
	let cases = [
		(0.1, 0.0, 0.0),
		(1.0, 0.0, 0.0),
		(10.0, 0.0, 0.0),
		(0.1, 0.1, 0.009950166250831947),
		(1.0, 0.1, 0.09516258196404043),
		(10.0, 0.1, 0.6321205588285577),
		(f64::INFINITY, 0.1, 1.0),
		(0.1, 1.0, 0.09516258196404043),
		(1.0, 1.0, 0.6321205588285577),
		(10.0, 1.0, 0.9999546000702375),
		(f64::INFINITY, 1.0, 1.0),
		(0.1, f64::INFINITY, 1.0),
		(1.0, f64::INFINITY, 1.0),
		(10.0, f64::INFINITY, 1.0),
		(f64::INFINITY, f64::INFINITY, 1.0),
		(0.1, -1.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.cdf(p_val), expected);
	}

	assert!(new_dist(f64::INFINITY).cdf(0.0).is_nan());
}

#[test]
fn test_inverse_cdf_identity() {
	let args = [0.42, 0.042, 0.0042, 0.33, 0.033, 0.0033];
	for rate in args {
		let dist = new_dist(rate);
		assert_close(
			rate,
			0.5,
			|d, p| d.inverse_cdf(p),
			dist.median().unwrap(),
		);
	}
}

#[test]
fn test_sf() {
	let cases = [
		(0.1, 0.0, 1.0),
		(1.0, 0.0, 1.0),
		(10.0, 0.0, 1.0),
		(0.1, 0.1, 0.9900498337491681),
		(1.0, 0.1, 0.9048374180359595),
		(10.0, 0.1, 0.36787944117144233),
		(f64::INFINITY, 0.1, 0.0),
		(0.1, -1.0, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.sf(p_val), expected);
	}

	assert!(new_dist(f64::INFINITY).sf(0.0).is_nan());
}

#[test]
fn test_continuous() {
	check_continuous_distribution(&new_dist(0.5), 0.0, 10.0);
	check_continuous_distribution(&new_dist(1.5), 0.0, 20.0);
	check_continuous_distribution(&new_dist(2.5), 0.0, 50.0);
}
