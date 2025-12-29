use stats::distribution::{Gamma, GammaError};

use crate::prelude::*;

make_test_harness!(Gamma(shape: f64, rate: f64), GammaError);

#[test]
fn test_new_is_ok() {
	let cases = [
		(1.0, 0.1),
		(1.0, 1.0),
		(10.0, 10.0),
		(10.0, 1.0),
		(10.0, f64::INFINITY),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((0.0, 0.0), GammaError::ShapeInvalid),
		((1.0, f64::NAN), GammaError::RateInvalid),
		((1.0, -1.0), GammaError::RateInvalid),
		((-1.0, 1.0), GammaError::ShapeInvalid),
		((-1.0, -1.0), GammaError::ShapeInvalid),
		((-1.0, f64::NAN), GammaError::ShapeInvalid),
		(
			(f64::INFINITY, f64::INFINITY),
			GammaError::ShapeAndRateInfinite,
		),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((1.0, 0.1), 10.0),
		((1.0, 1.0), 1.0),
		((10.0, 10.0), 1.0),
		((10.0, 1.0), 10.0),
		((10.0, f64::INFINITY), 0.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((1.0, 0.1), 100.0),
		((1.0, 1.0), 1.0),
		((10.0, 10.0), 0.1),
		((10.0, 1.0), 10.0),
		((10.0, f64::INFINITY), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((1.0, 0.1), 3.3025850929940455),
		((1.0, 1.0), 1.0),
		((10.0, 10.0), 0.23346908548693396),
		((10.0, 1.0), 2.53605417848098),
		((10.0, f64::INFINITY), f64::NEG_INFINITY),
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
		((10.0, 10.0), 0.6324555320336759),
		((10.0, 1.0), 0.6324555320336759),
		((10.0, f64::INFINITY), 0.6324555320336758),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((1.0, 0.1), 0.0),
		((1.0, 1.0), 0.0),
		((10.0, 10.0), 0.9),
		((10.0, 1.0), 9.0),
		((10.0, f64::INFINITY), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [
		((1.0, 0.1), 0.0),
		((1.0, 1.0), 0.0),
		((10.0, 10.0), 0.0),
		((10.0, 1.0), 0.0),
		((10.0, f64::INFINITY), 0.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [
		((1.0, 0.1), f64::INFINITY),
		((1.0, 1.0), f64::INFINITY),
		((10.0, 10.0), f64::INFINITY),
		((10.0, 1.0), f64::INFINITY),
		((10.0, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((1.0, 0.1), 1.0, 0.09048374180359596),
		((1.0, 0.1), 10.0, 0.036787944117144235),
		((1.0, 1.0), 1.0, 0.36787944117144233),
		((1.0, 1.0), 10.0, 0.000045399929762484854),
		((10.0, 10.0), 1.0, 1.251100357211333),
		((10.0, 10.0), 10.0, 1.0251532120868705e-30),
		((10.0, 1.0), 1.0, 0.0000010137771196302974),
		((10.0, 1.0), 10.0, 0.1251100357211333),
		// at zero
		((1.0, 0.1), 0.0, 0.1),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pdf(p), expected);
	}
	// TODO: test special
	// is this really the behavior we want?
	// test_is_nan((10.0, f64::INFINITY), pdf(1.0));
	// TODO: test special
	// (10.0, f64::INFINITY, f64::INFINITY, 0.0, pdf(f64::INFINITY)),];
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((1.0, 0.1), 1.0, -2.4025850929940455),
		((1.0, 0.1), 10.0, -3.3025850929940455),
		((1.0, 1.0), 1.0, -1.0),
		((1.0, 1.0), 10.0, -10.0),
		((10.0, 10.0), 1.0, 0.22402344985898723),
		((10.0, 10.0), 10.0, -69.0527107131946),
		((10.0, 1.0), 1.0, -13.801827480081469),
		((10.0, 1.0), 10.0, -2.0785616431350586),
		((10.0, f64::INFINITY), f64::INFINITY, f64::NEG_INFINITY),
		// at zero
		((1.0, 0.1), 0.0, 0.1.ln()),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}
	// TODO: test special
	// is this really the behavior we want?
	// test_is_nan((10.0, f64::INFINITY), f(1.0));
}

#[test]
fn test_cdf() {
	let cases = [
		((1.0, 0.1), 1.0, 0.09516258196404043),
		((1.0, 0.1), 10.0, 0.6321205588285577),
		((1.0, 1.0), 1.0, 0.6321205588285577),
		((1.0, 1.0), 10.0, 0.9999546000702375),
		((10.0, 10.0), 1.0, 0.5420702855281478),
		((10.0, 10.0), 10.0, 0.999999999999999999999999),
		((10.0, 1.0), 1.0, 0.00000011142547833872067),
		((10.0, 1.0), 10.0, 0.5420702855281478),
		((10.0, f64::INFINITY), 1.0, 0.0),
		((10.0, f64::INFINITY), 10.0, 1.0),
		// at zero
		((1.0, 0.1), 0.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_cdf_inverse_identity() {
	let args = [
		(1.0, 0.1),
		(1.0, 1.0),
		(10.0, 10.0),
		(10.0, 1.0),
		(100.0, 200.0),
	];

	for args in args {
		for n in -5..0 {
			let p = 10.0f64.powi(n);
			let p = Probability::new(p).unwrap();
			assert_close(
				args,
				p,
				|d, p| d.cdf(d.inverse_cdf(p)),
				*p,
			);
		}
	}

	// https://github.com/statrs-dev/statrs/issues/200
	let args = (3.0, 0.5);
	let p = 20.5567;
	// TODO: this fails for (1.0, 1.0) too
	assert_close(
		args,
		p,
		|d, p| {
			let prob = Probability::new(d.cdf(p)).unwrap();
			d.inverse_cdf(prob)
		},
		p,
	);
}

#[test]
fn test_sf() {
	let cases = [
		((1.0, 0.1), 1.0, 0.9048374180359595),
		((1.0, 0.1), 10.0, 0.3678794411714419),
		((1.0, 1.0), 1.0, 0.3678794411714419),
		((1.0, 1.0), 10.0, 4.539992976249074e-5),
		((10.0, 10.0), 1.0, 0.4579297144718528),
		((10.0, 10.0), 10.0, 1.1253473960842808e-31),
		((10.0, 1.0), 1.0, 0.9999998885745217),
		((10.0, 1.0), 10.0, 0.4579297144718528),
		((10.0, f64::INFINITY), 1.0, 1.0),
		((10.0, f64::INFINITY), 10.0, 0.0),
		// at zero
		((1.0, 0.1), 0.0, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_continuous() {
	check_continuous_distribution(&new_dist((1.0, 0.5)), 0.0, 20.0);
	check_continuous_distribution(&new_dist((9.0, 2.0)), 0.0, 20.0);
}
