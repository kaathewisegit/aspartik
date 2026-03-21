use stats::distribution::{Normal, NormalError};

// TODO: precision
use math::assert_almost_eq;

use crate::prelude::*;

make_test_harness!(Normal(mean: f64, std_dev: f64), NormalError);

#[test]
fn test_new_is_ok() {
	let cases = [
		(10.0, 0.1),
		(-5.0, 1.0),
		(0.0, 10.0),
		(10.0, 100.0),
		(-5.0, f64::INFINITY),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((f64::NAN, 1.0), NormalError::MeanInvalid),
		((1.0, f64::NAN), NormalError::StandardDeviationInvalid),
		((0.0, 0.0), NormalError::StandardDeviationInvalid),
		((f64::NAN, f64::NAN), NormalError::MeanInvalid),
		((1.0, -1.0), NormalError::StandardDeviationInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((0.0, 0.1), 0.1 * 0.1),
		((0.0, 1.0), 1.0),
		((0.0, 10.0), 100.0),
		((0.0, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((0.0, 0.1), -0.8836465597893729),
		((0.0, 1.0), 1.4189385332046727),
		((0.0, 10.0), 3.7215236261987186),
		((0.0, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((0.0, 0.1), 0.0),
		((4.0, 1.0), 0.0),
		((0.3, 10.0), 0.0),
		((0.0, f64::INFINITY), 0.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((-0.0, 1.0), 0.0),
		((0.0, 1.0), 0.0),
		((0.1, 1.0), 0.1),
		((1.0, 1.0), 1.0),
		((-10.0, 1.0), -10.0),
		((f64::INFINITY, 1.0), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((-0.0, 1.0), 0.0),
		((0.0, 1.0), 0.0),
		((0.1, 1.0), 0.1),
		((1.0, 1.0), 1.0),
		((-0.0, 1.0), -0.0),
		((f64::INFINITY, 1.0), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [
		((0.0, 0.1), f64::NEG_INFINITY),
		((-3.0, 10.0), f64::NEG_INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases =
		[((0.0, 0.1), f64::INFINITY), ((-3.0, 10.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((10.0, 0.1), 8.5, 5.530709549844416E-49),
		((10.0, 0.1), 9.8, 0.5399096651318805),
		((10.0, 0.1), 10.0, 3.989422804014327),
		((10.0, 0.1), 10.2, 0.5399096651318805),
		((10.0, 0.1), 11.5, 5.530709549844416E-49),
		((-5.0, 1.0), -10.0, 1.4867195147342977E-6),
		((-5.0, 1.0), -7.5, 0.017528300493568537),
		((-5.0, 1.0), -5.0, 0.3989422804014327),
		((-5.0, 1.0), -2.5, 0.017528300493568537),
		((-5.0, 1.0), 0.0, 1.4867195147342977E-6),
		((0.0, 10.0), -5.0, 0.035206532676429945),
		((0.0, 10.0), -2.5, 0.03866681168028492),
		((0.0, 10.0), 0.0, 0.03989422804014327),
		((0.0, 10.0), 2.5, 0.03866681168028492),
		((0.0, 10.0), 5.0, 0.035206532676429945),
		((10.0, 100.0), -200.0, 4.3983595980427194E-4),
		((10.0, 100.0), -100.0, 0.0021785217703255053),
		((10.0, 100.0), 0.0, 0.003969525474770118),
		((10.0, 100.0), 100.0, 0.0026608524989875483),
		((10.0, 100.0), 200.0, 6.561581477467659E-4),
		((-5.0, f64::INFINITY), -5.0, 0.0),
		((-5.0, f64::INFINITY), 0.0, 0.0),
		((-5.0, f64::INFINITY), 100.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((10.0, 0.1), 8.5, 5.530709549844416E-49_f64.ln()),
		((10.0, 0.1), 9.8, 0.5399096651318805_f64.ln()),
		((10.0, 0.1), 10.0, 3.989422804014327_f64.ln()),
		((10.0, 0.1), 10.2, 0.5399096651318805_f64.ln()),
		((10.0, 0.1), 11.5, 5.530709549844416E-49_f64.ln()),
		((-5.0, 1.0), -10.0, 1.4867195147342977E-6_f64.ln()),
		((-5.0, 1.0), -7.5, 0.017528300493568537_f64.ln()),
		((-5.0, 1.0), -5.0, 0.3989422804014327_f64.ln()),
		((-5.0, 1.0), -2.5, 0.017528300493568537_f64.ln()),
		((-5.0, 1.0), 0.0, 1.4867195147342977E-6_f64.ln()),
		((0.0, 10.0), -5.0, 0.035206532676429945_f64.ln()),
		((0.0, 10.0), -2.5, 0.03866681168028492_f64.ln()),
		((0.0, 10.0), 0.0, 0.03989422804014327_f64.ln()),
		((0.0, 10.0), 2.5, 0.03866681168028492_f64.ln()),
		((0.0, 10.0), 5.0, 0.035206532676429945_f64.ln()),
		((10.0, 100.0), -200.0, 4.3983595980427194E-4_f64.ln()),
		((10.0, 100.0), -100.0, 0.0021785217703255053_f64.ln()),
		((10.0, 100.0), 0.0, 0.003969525474770118_f64.ln()),
		((10.0, 100.0), 100.0, 0.0026608524989875483_f64.ln()),
		((10.0, 100.0), 200.0, 6.561581477467659E-4_f64.ln()),
		((-5.0, f64::INFINITY), -5.0, f64::NEG_INFINITY),
		((-5.0, f64::INFINITY), 0.0, f64::NEG_INFINITY),
		((-5.0, f64::INFINITY), 100.0, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((5.0, 2.0), f64::NEG_INFINITY, 0.0),
		((5.0, 2.0), -5.0, 0.0000002866515718),
		((5.0, 2.0), -2.0, 0.0002326290790),
		((5.0, 2.0), 0.0, 0.006209665325),
		((5.0, 2.0), 4.0, 0.3085375387259869),
		((5.0, 2.0), 5.0, 0.5),
		((5.0, 2.0), 6.0, 0.6914624612740131),
		((5.0, 2.0), 10.0, 0.993790334674),
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		assert_almost_eq!(dist.cdf(p), expected, epsilon = 1e-12);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((5.0, 2.0), f64::NEG_INFINITY, 1.0),
		((5.0, 2.0), -5.0, 0.9999997133484281),
		((5.0, 2.0), -2.0, 0.9997673709209455),
		((5.0, 2.0), 0.0, 0.9937903346744879),
		((5.0, 2.0), 4.0, 0.6914624612740131),
		((5.0, 2.0), 5.0, 0.5),
		((5.0, 2.0), 6.0, 0.3085375387259869),
		((5.0, 2.0), 10.0, 0.006209665325512148),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_continuous() {
	let cases = [((0.0, 1.0), -10.0, 10.0), ((20.0, 0.5), 10.0, 30.0)];
	for (args, lower, upper) in cases {
		check_continuous_distribution(&new_dist(args), lower, upper);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((5.0, 2.0), 0.0, f64::NEG_INFINITY),
		((5.0, 2.0), 0.0000002866515718791939, -5.0),
		((5.0, 2.0), 0.00023262907903552504, -2.0),
		((5.0, 2.0), 0.006209665325776135, -0.0),
		((5.0, 2.0), 0.006209665325776135, 0.0),
		((5.0, 2.0), 0.3085375387259869, 4.0),
		((5.0, 2.0), 0.5, 5.0),
		((5.0, 2.0), 0.6914624612740131, 6.0),
		((5.0, 2.0), 0.9937903346742238, 10.0),
		((5.0, 2.0), 1.0, f64::INFINITY),
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		let p = Probability::new(p);
		assert_almost_eq!(
			dist.inverse_cdf(p),
			expected,
			epsilon = 1e-14,
		);
	}
}

#[test]
fn test_default() {
	let n = Normal::default();

	let n_mean = n.mean().unwrap();
	let n_std = n.std_dev().unwrap();

	assert_eq!(n_mean, 0.0);
	assert_eq!(n_std, 1.0);
}
