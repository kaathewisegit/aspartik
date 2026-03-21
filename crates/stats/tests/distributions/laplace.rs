use stats::distribution::{Laplace, LaplaceError};

use crate::prelude::*;
use std::f64::consts::{E, LN_2, LN_10};

make_test_harness!(Laplace(location: f64, scale: f64), LaplaceError);

#[test]
fn test_new_is_ok() {
	let cases = [
		(1.0, 2.0),
		(f64::NEG_INFINITY, 0.1),
		(-5.0 - 1.0, 1.0),
		(0.0, 5.0),
		(1.0, 7.0),
		(5.0, 10.0),
		(f64::INFINITY, f64::INFINITY),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((2.0, -1.0), LaplaceError::ScaleInvalid),
		((f64::NAN, 1.0), LaplaceError::LocationInvalid),
		((f64::NAN, -1.0), LaplaceError::LocationInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((f64::NEG_INFINITY, 0.1), f64::NEG_INFINITY),
		((-6.0, 1.0), -6.0),
		((0.0, 5.0), 0.0),
		((1.0, 10.0), 1.0),
		((f64::INFINITY, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((f64::NEG_INFINITY, 0.1), 0.02),
		((-6.0, 1.0), 2.0),
		((0.0, 5.0), 50.0),
		((1.0, 7.0), 98.0),
		((5.0, 10.0), 200.0),
		((f64::INFINITY, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((f64::NEG_INFINITY, 0.1), (2.0 * E * 0.1).ln()),
		((-6.0, 1.0), (2.0 * E).ln()),
		((1.0, 7.0), (2.0 * E * 7.0).ln()),
		((5.0, 10.0), (2.0 * E * 10.0).ln()),
		((f64::INFINITY, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((f64::NEG_INFINITY, 0.1), 0.0),
		((-6.0, 1.0), 0.0),
		((1.0, 7.0), 0.0),
		((5.0, 10.0), 0.0),
		((f64::INFINITY, f64::INFINITY), 0.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((f64::NEG_INFINITY, 0.1), f64::NEG_INFINITY),
		((-6.0, 1.0), -6.0),
		((1.0, 7.0), 1.0),
		((5.0, 10.0), 5.0),
		((f64::INFINITY, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((f64::NEG_INFINITY, 0.1), f64::NEG_INFINITY),
		((-6.0, 1.0), -6.0),
		((1.0, 7.0), 1.0),
		((5.0, 10.0), 5.0),
		((f64::INFINITY, f64::INFINITY), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
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
fn test_pdf() {
	let cases = [
		((0.0, 0.1), 1.5, 1.529511602509129e-6),
		((1.0, 0.1), 2.8, 7.614989872356341e-8),
		((-1.0, 0.1), -5.4, 3.8905661205668983e-19),
		((5.0, 0.1), -4.9, 5.056107463052243e-43),
		((-5.0, 0.1), 2.0, 1.9877248679543235e-30),
		((f64::INFINITY, 0.1), 5.5, 0.0),
		((f64::NEG_INFINITY, 0.1), -0.0, 0.0),
		((0.0, 1.0), f64::INFINITY, 0.0),
		((1.0, 1.0), 5.0, 0.00915781944436709),
		((-1.0, 1.0), -1.0, 0.5),
		((5.0, 1.0), -1.0, 0.0012393760883331792),
		((-5.0, 1.0), 2.5, 0.0002765421850739168),
		((f64::INFINITY, 0.1), 2.0, 0.0),
		((f64::NEG_INFINITY, 0.1), 15.0, 0.0),
		((0.0, f64::INFINITY), 89.3, 0.0),
		((1.0, f64::INFINITY), -0.1, 0.0),
		((-1.0, f64::INFINITY), 0.1, 0.0),
		((5.0, f64::INFINITY), -6.1, 0.0),
		((-5.0, f64::INFINITY), -10.0, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pdf(p), expected);
	}

	let dist = new_dist((f64::INFINITY, f64::INFINITY));
	for p in [2.0, -5.1] {
		assert!(dist.pdf(p).is_nan());
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((0.0, 0.1), 1.5, -13.3905620875659),
		((1.0, 0.1), 2.8, -16.390562087565897),
		((-1.0, 0.1), -5.4, -42.39056208756591),
		((5.0, 0.1), -4.9, -97.3905620875659),
		((-5.0, 0.1), 2.0, -68.3905620875659),
		((f64::INFINITY, 0.1), 5.5, f64::NEG_INFINITY),
		((f64::NEG_INFINITY, 0.1), -0.0, f64::NEG_INFINITY),
		((0.0, 1.0), f64::INFINITY, f64::NEG_INFINITY),
		((1.0, 1.0), 5.0, -4.693147180559945),
		((-1.0, 1.0), -1.0, -LN_2),
		((5.0, 1.0), -1.0, -6.693147180559945),
		((-5.0, 1.0), 2.5, -8.193147180559945),
		((f64::INFINITY, 0.1), 2.0, f64::NEG_INFINITY),
		((f64::NEG_INFINITY, 0.1), 15.0, f64::NEG_INFINITY),
		((0.0, f64::INFINITY), 89.3, f64::NEG_INFINITY),
		((1.0, f64::INFINITY), -0.1, f64::NEG_INFINITY),
		((-1.0, f64::INFINITY), 0.1, f64::NEG_INFINITY),
		((5.0, f64::INFINITY), -6.1, f64::NEG_INFINITY),
		((-5.0, f64::INFINITY), -10.0, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}

	let dist = new_dist((f64::INFINITY, f64::INFINITY));
	for p in [2.0, -5.1] {
		assert!(dist.ln_pdf(p).is_nan());
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.0, 1.0), 0.5, 0.6967346701436833),
		((0.0, 1.0), -0.5, 0.3032653298563167),
		((0.0, 1.0), -100.0, 1.860037988010418e-44),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.0, 1.0), 0.5, 0.3032653298563167),
		((0.0, 1.0), -0.5, 0.6967346701436833),
		((0.0, 1.0), 100.0, 1.860037988010418e-44),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0.0, 1.0), 1e-10, -22.33270374938051),
		((0.0, 1.0), 0.001, -6.214608098422191),
		((0.0, 1.0), 0.95, LN_10),
	];
	for (args, p, expected) in cases {
		let p = Probability::new(p);
		assert_close(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}

#[cfg(feature = "rand")]
#[test]
fn test_sample() {
	use rand::distr::Distribution;
	use rand::rng;

	let l = new_dist((0.1, 0.5));
	l.sample(&mut rng());
}

#[cfg(feature = "rand")]
#[test]
fn test_sample_distribution() {
	use rand::SeedableRng;
	use rand::distr::Distribution;
	use rand::rngs::StdRng;

	let location = 0.0;
	let scale = 1.0;
	let n = new_dist((location, scale));
	let trials = 10_000;
	let tolerance = 250;

	for seed in 0..10 {
		let mut r: StdRng = SeedableRng::seed_from_u64(seed);

		let result = (0..trials).map(|_| n.sample(&mut r)).fold(
			0,
			|sum, val| {
				if val > 0.0 {
					sum + 1
				} else if val < 0.0 {
					sum - 1
				} else {
					0
				}
			},
		);
		assert!(
			result > -tolerance && result < tolerance,
			"Balance is {result} for seed {seed}"
		);
	}
}
