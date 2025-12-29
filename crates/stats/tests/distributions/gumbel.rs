use stats::distribution::{Gumbel, GumbelError};

use crate::prelude::*;

make_test_harness!(Gumbel(location: f64, scale: f64), GumbelError);

#[test]
fn test_new_is_ok() {
	let cases = [
		(0.0, 0.1),
		(0.0, 1.0),
		(0.0, 10.0),
		(10.0, 11.0),
		(-5.0, 100.0),
		(0.0, f64::INFINITY),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((f64::NAN, 1.0), GumbelError::LocationInvalid),
		((1.0, f64::NAN), GumbelError::ScaleInvalid),
		((f64::NAN, f64::NAN), GumbelError::LocationInvalid),
		((1.0, 0.0), GumbelError::ScaleInvalid),
		((0.0, f64::NEG_INFINITY), GumbelError::ScaleInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_lower() {
	let cases = [((0.0, 1.0), f64::NEG_INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((0.0, 1.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((0.0, 2.0), 2.270362845461478),
		((0.1, 4.0), 2.9635100260214235),
		((1.0, 10.0), 3.8798007578955787),
		((10.0, 11.0), 3.9751109376999034),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((0.0, 2.0), 1.1544313298030658),
		((0.1, 4.0), 2.4088626596061316),
		((1.0, 10.0), 6.772156649015328),
		((10.0, 11.0), 16.34937231391686),
		((10.0, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((0.0, 2.0), 1.13955),
		((0.1, 4.0), 1.13955),
		((1.0, 10.0), 1.13955),
		((10.0, 11.0), 1.13955),
		((10.0, f64::INFINITY), 1.13955),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((0.0, 2.0), 6.579736267392906),
		((0.1, 4.0), 26.318945069571624),
		((1.0, 10.0), 164.49340668482265),
		((10.0, 11.0), 199.03702208863538),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_std_dev() {
	let cases = [
		((0.0, 2.0), 2.565099660323728),
		((0.1, 4.0), 5.130199320647456),
		((1.0, 10.0), 12.82549830161864),
		((10.0, 11.0), 14.108048131780505),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.std_dev().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((0.0, 2.0), 0.7330258411633287),
		((0.1, 4.0), 1.5660516823266574),
		((1.0, 10.0), 4.665129205816644),
		((10.0, 11.0), 14.031642126398307),
		((10.0, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((0.0, 2.0), 0.0),
		((0.1, 4.0), 0.1),
		((1.0, 10.0), 1.0),
		((10.0, 11.0), 10.0),
		((10.0, f64::INFINITY), 10.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mode(), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.0, 0.1), -5.0, 0.0),
		((0.0, 0.1), -1.0, 0.0),
		((0.0, 0.1), 0.0, 0.36787944117144233),
		((0.0, 0.1), 1.0, 0.9999546011007987),
		((0.0, 0.1), 5.0, 0.99999999999999999),
		((0.0, 1.0), -1.0, 0.06598803584531253),
		((0.0, 1.0), 0.0, 0.36787944117144233),
		((0.0, 10.0), -5.0, 0.19229564554796494),
		((0.0, 10.0), -1.0, 0.3311542771529088),
		((0.0, 10.0), 0.0, 0.36787944117144233),
		((0.0, 10.0), 1.0, 0.4046076616641318),
		((0.0, 10.0), 5.0, 0.545239211892605),
		((-2.0, f64::INFINITY), -5.0, 0.36787944117144233),
		((-2.0, f64::INFINITY), -1.0, 0.36787944117144233),
		((-2.0, f64::INFINITY), 0.0, 0.36787944117144233),
		((-2.0, f64::INFINITY), 1.0, 0.36787944117144233),
		((-2.0, f64::INFINITY), 5.0, 0.36787944117144233),
		((f64::INFINITY, 1.0), -5.0, 0.0),
		((f64::INFINITY, 1.0), -1.0, 0.0),
		((f64::INFINITY, 1.0), 0.0, 0.0),
		((f64::INFINITY, 1.0), 1.0, 0.0),
		((f64::INFINITY, 1.0), 5.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.cdf(p_val), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0.0, 0.1), 0.0, f64::NEG_INFINITY),
		((0.0, 0.1), 1.0, f64::INFINITY),
		((0.0, 1.0), 0.1, -0.8340324452479557),
		((0.0, 10.0), 0.5, 3.6651292058166436),
		((0.0, 10.0), 0.9, 22.503673273124456),
		((2.0, f64::INFINITY), 0.1, f64::NEG_INFINITY),
		((-2.0, f64::INFINITY), 0.5, f64::INFINITY),
		((f64::INFINITY, 1.0), 0.1, f64::INFINITY),
	];
	for (args, p, expected) in cases {
		let p = Probability::new(p).unwrap();
		assert_close(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.0, 0.1), -5.0, 1.0),
		((0.0, 0.1), -1.0, 1.0),
		((0.0, 0.1), 0.0, 0.6321205588285577),
		((0.0, 0.1), 1.0, 0.000045398899201269),
		((0.0, 1.0), -1.0, 0.9340119641546875),
		((0.0, 1.0), 0.0, 0.6321205588285577),
		((0.0, 1.0), 1.0, 0.3077993724446536),
		((0.0, 10.0), -1.0, 0.6688457228470911),
		((0.0, 10.0), 0.0, 0.6321205588285577),
		((0.0, 10.0), 1.0, 0.5953923383358681),
		((-2.0, f64::INFINITY), -5.0, 0.6321205588285577),
		((-2.0, f64::INFINITY), -1.0, 0.6321205588285577),
		((-2.0, f64::INFINITY), 0.0, 0.6321205588285577),
		((-2.0, f64::INFINITY), 1.0, 0.6321205588285577),
		((-2.0, f64::INFINITY), 5.0, 0.6321205588285577),
		((f64::INFINITY, 1.0), -5.0, 1.0),
		((f64::INFINITY, 1.0), -1.0, 1.0),
		((f64::INFINITY, 1.0), 0.0, 1.0),
		((f64::INFINITY, 1.0), 1.0, 1.0),
		((f64::INFINITY, 1.0), 5.0, 1.0),
		((0.0, 1.0), 40.0, 4.248354255291589e-18),
		((0.0, 1.0), 80.0, 1.804851387845415e-35),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.sf(p_val), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((0.0, 0.1), -5.0, 0.0),
		((0.0, 0.1), 0.0, 3.6787944117144233),
		((0.0, 0.1), 1.0, 0.0004539786865564),
		((0.0, 1.0), -1.0, 0.1793740787340171),
		((0.0, 1.0), 0.0, 0.36787944117144233),
		((0.0, 1.0), 1.0, 0.2546463800435825),
		((0.0, 10.0), -5.0, 0.03170419210779422),
		((0.0, 10.0), -1.0, 0.0365982076505757),
		((0.0, 10.0), 0.0, 0.036787944117144233),
		((0.0, 10.0), 1.0, 0.036610415189774016),
		((0.0, 10.0), 5.0, 0.033070429889041),
		((-2.0, f64::INFINITY), -5.0, 0.0),
		((-2.0, f64::INFINITY), -1.0, 0.0),
		((-2.0, f64::INFINITY), 0.0, 0.0),
		((-2.0, f64::INFINITY), 1.0, 0.0),
		((-2.0, f64::INFINITY), 5.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.pdf(p_val), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((0.0, 0.1), -5.0, f64::NEG_INFINITY),
		((0.0, 0.1), 0.0, 3.6787944117144233_f64.ln()),
		((0.0, 0.1), 1.0, 0.0004539786865564_f64.ln()),
		((0.0, 1.0), -1.0, 0.1793740787340171_f64.ln()),
		((0.0, 1.0), 0.0, 0.36787944117144233_f64.ln()),
		((0.0, 1.0), 1.0, 0.2546463800435825_f64.ln()),
		((0.0, 10.0), -5.0, 0.03170419210779422_f64.ln()),
		((0.0, 10.0), -1.0, 0.0365982076505757_f64.ln()),
		((0.0, 10.0), 0.0, 0.036787944117144233_f64.ln()),
		((0.0, 10.0), 1.0, 0.036610415189774016_f64.ln()),
		((0.0, 10.0), 5.0, 0.033070429889041_f64.ln()),
		((-2.0, f64::INFINITY), -5.0, f64::NEG_INFINITY),
		((-2.0, f64::INFINITY), -1.0, f64::NEG_INFINITY),
		((-2.0, f64::INFINITY), 0.0, f64::NEG_INFINITY),
		((-2.0, f64::INFINITY), 1.0, f64::NEG_INFINITY),
		((-2.0, f64::INFINITY), 5.0, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p_val| d.ln_pdf(p_val), expected);
	}
}

#[test]
fn test_cdf_inverse_identity() {
	let args = [
		(0.0, 0.1),
		(0.0, 1.0),
		(0.0, 10.0),
		(10.0, 11.0),
		(-5.0, 100.0),
	];

	for args in args {
		// Test values within (0, 1) for cdf and inverse_cdf identity
		let test_points = [0.1, 0.5, 0.9];
		for p in test_points {
			let p = Probability::new(p).unwrap();
			assert_close(
				args,
				p,
				|d, p| d.cdf(d.inverse_cdf(p)),
				*p,
			);
		}
	}
}

// TODO: check_continuous_distribution, currently returns NaN for -inf
