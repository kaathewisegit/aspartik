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
		((1.0, 0.1), 3.302585092994045628506840223),
		((1.0, 1.0), 1.0),
		((10.0, 10.0), 0.2334690854869339583626209),
		((10.0, 1.0), 2.53605417848097964238061239),
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
		((10.0, 10.0), 0.6324555320336758663997787),
		((10.0, 1.0), 0.63245553203367586639977870),
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
		((1.0, 0.1), 1.0, 0.090483741803595961836995),
		((1.0, 0.1), 10.0, 0.036787944117144234201693),
		((1.0, 1.0), 1.0, 0.367879441171442321595523),
		((1.0, 1.0), 10.0, 0.000045399929762484851535),
		((10.0, 10.0), 1.0, 1.251100357211332989847649),
		((10.0, 10.0), 10.0, 1.025153212086870580621609e-30),
		((10.0, 1.0), 1.0, 0.000001013777119630297402),
		((10.0, 1.0), 10.0, 0.125110035721133298984764),
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
		((1.0, 0.1), 1.0, -2.40258509299404563405795),
		((1.0, 0.1), 10.0, -3.30258509299404562850684),
		((1.0, 1.0), 1.0, -1.0),
		((1.0, 1.0), 10.0, -10.0),
		((10.0, 10.0), 1.0, 0.224023449858987228972196),
		((10.0, 10.0), 10.0, -69.0527107131946016148658),
		((10.0, 1.0), 1.0, -13.8018274800814696112077),
		((10.0, 1.0), 10.0, -2.07856164313505845504579),
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
		((1.0, 0.1), 1.0, 0.095162581964040431858607),
		((1.0, 0.1), 10.0, 0.632120558828557678404476),
		((1.0, 1.0), 1.0, 0.632120558828557678404476),
		((1.0, 1.0), 10.0, 0.999954600070237515148464),
		((10.0, 10.0), 1.0, 0.542070285528147791685835),
		((10.0, 10.0), 10.0, 0.999999999999999999999999),
		((10.0, 1.0), 1.0, 0.000000111425478338720677),
		((10.0, 1.0), 10.0, 0.542070285528147791685835),
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
			assert_close(
				args,
				p,
				|d, p| d.cdf(d.inverse_cdf(p)),
				p,
			);
		}
	}

	// https://github.com/statrs-dev/statrs/issues/200
	let args = (3.0, 0.5);
	let p = 20.5567;
	// TODO: this fails for (1.0, 1.0) too
	assert_close(args, p, |d, p| d.inverse_cdf(d.cdf(p)), p);
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
	check_continuous_distribution(
		&Gamma::new(1.0, 0.5).unwrap(),
		0.0,
		20.0,
	);
	check_continuous_distribution(
		&Gamma::new(9.0, 2.0).unwrap(),
		0.0,
		20.0,
	);
}
