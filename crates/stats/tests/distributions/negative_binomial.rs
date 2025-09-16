use stats::distribution::{NegativeBinomial, NegativeBinomialError};

use math::assert_almost_eq;

use crate::prelude::*;

make_test_harness!(
    NegativeBinomial(r: f64, p: f64),
    NegativeBinomialError
);

#[test]
fn test_new_is_ok() {
	let cases = [(0.0, 0.0), (0.3, 0.4), (1.0, 0.3)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((f64::NAN, 1.0), NegativeBinomialError::RInvalid),
		((0.0, f64::NAN), NegativeBinomialError::PInvalid),
		((-1.0, 1.0), NegativeBinomialError::RInvalid),
		((2.0, 2.0), NegativeBinomialError::PInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((4.0, 0.0), f64::INFINITY),
		((3.0, 0.3), 7.0),
		((2.0, 1.0), 0.0),
	];
	for (args, expected) in cases {
		let dist = new_dist(args);
		assert_almost_eq!(
			dist.mean().unwrap(),
			expected,
			epsilon = 1e-15,
		);
	}
}

#[test]
fn test_variance() {
	let cases = [
		((4.0, 0.0), f64::INFINITY),
		((3.0, 0.3), 23.333333333333),
		((2.0, 1.0), 0.0),
	];
	for (args, expected) in cases {
		let dist = new_dist(args);
		assert_almost_eq!(
			dist.variance().unwrap(),
			expected,
			epsilon = 1e-12,
		);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((0.0, 0.0), f64::INFINITY),
		((0.1, 0.3), 6.425396041),
		((1.0, 1.0), f64::INFINITY),
	];
	for (args, expected) in cases {
		let dist = new_dist(args);
		assert_almost_eq!(
			dist.skewness().unwrap(),
			expected,
			epsilon = 1e-9,
		);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((0.0, 0.0), 0.0),
		((0.3, 0.0), 0.0),
		((1.0, 1.0), 0.0),
		((10.0, 0.01), 891.0),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [((1.0, 0.5), 0), ((1.0, 0.3), 0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((1.0, 0.5), u64::MAX), ((1.0, 0.3), u64::MAX)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pmf() {
	let cases = [
		((4.0, 0.5), 0, 0.0625),
		((4.0, 0.5), 3, 0.15625),
		((1.0, 0.0), 0, 0.0),
		((1.0, 0.0), 1, 0.0),
		((3.0, 0.2), 0, 0.008),
		((3.0, 0.2), 1, 0.0192),
		((3.0, 0.2), 3, 0.04096),
		((10.0, 0.2), 0, 1.024e-07),
		((10.0, 0.2), 1, 8.192e-07),
		((10.0, 0.2), 10, 0.001015706852),
		((1.0, 0.3), 0, 0.3),
		((1.0, 0.3), 1, 0.21),
		((3.0, 0.3), 0, 0.027),
		((0.3, 1.0), 1, 0.0),
		((0.3, 1.0), 3, 0.0),
		((0.3, 1.0), 1, 0.0),
		((0.3, 1.0), 10, 0.0),
		((1.0, 1.0), 1, 0.0),
		((3.0, 1.0), 1, 0.0),
		((3.0, 1.0), 3, 0.0),
		((10.0, 1.0), 1, 0.0),
		((10.0, 1.0), 10, 0.0),
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		assert_almost_eq!(dist.pmf(p), expected, epsilon = 1e-12);
	}
}

#[test]
fn test_pmf_nan() {
	let cases = [((0.3, 1.0), 0), ((3.0, 1.0), 0), ((10.0, 1.0), 0)];
	for (args, p) in cases {
		assert!(new_dist(args).pmf(p).is_nan());
	}
}

#[test]
fn test_ln_pmf() {
	let cases = [
		((1.0, 0.0), 0, f64::NEG_INFINITY),
		((1.0, 0.0), 1, f64::NEG_INFINITY),
		((3.0, 0.2), 0, -4.828313737),
		((3.0, 0.2), 1, -3.952845),
		((3.0, 0.2), 3, -3.195159298),
		((10.0, 0.2), 0, -16.09437912),
		((10.0, 0.2), 1, -14.01493758),
		((10.0, 0.2), 10, -6.892170503),
		((1.0, 0.3), 0, -1.203972804),
		((1.0, 0.3), 1, -1.560647748),
		((3.0, 0.3), 0, -3.611918413),
		((0.3, 1.0), 1, f64::NEG_INFINITY),
		((0.3, 1.0), 3, f64::NEG_INFINITY),
		((0.3, 1.0), 1, f64::NEG_INFINITY),
		((0.3, 1.0), 10, f64::NEG_INFINITY),
		((1.0, 1.0), 1, f64::NEG_INFINITY),
		((3.0, 1.0), 1, f64::NEG_INFINITY),
		((3.0, 1.0), 3, f64::NEG_INFINITY),
		((10.0, 1.0), 1, f64::NEG_INFINITY),
		((10.0, 1.0), 10, f64::NEG_INFINITY),
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		assert_almost_eq!(dist.ln_pmf(p), expected, epsilon = 1e-8);
	}
}

#[test]
fn test_ln_pmf_nan() {
	let cases = [
		((0.3, 1.0), 0),
		((1.0, 1.0), 0),
		((3.0, 1.0), 0),
		((10.0, 1.0), 0),
	];
	for (args, p) in cases {
		assert!(new_dist(args).pmf(p).is_nan());
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((1.0, 0.3), 0, 0.3),
		((1.0, 0.3), 1, 0.51),
		((1.0, 0.3), 4, 0.83193),
		((1.0, 0.3), 10, 0.9802267326),
		((1.0, 1.0), 0, 1.0),
		((1.0, 1.0), 1, 1.0),
		((10.0, 0.75), 0, 0.05631351471),
		((10.0, 0.75), 1, 0.1970973015),
		((10.0, 0.75), 10, 0.9960578583),
		((3.0, 0.5), 100, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((1.0, 0.3), 0, 0.7),
		((1.0, 0.3), 1, 0.49),
		((1.0, 0.3), 4, 0.1680699999999986),
		((1.0, 0.3), 10, 0.019773267430000074),
		((1.0, 1.0), 0, 0.0),
		((1.0, 1.0), 1, 0.0),
		((10.0, 0.75), 0, 0.9436864852905275),
		((10.0, 0.75), 1, 0.8029026985168456),
		((10.0, 0.75), 10, 0.003942141664083465),
		((3.0, 0.5), 100, 5.282409836586059e-28),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [((3.0, 0.5), 1.0, u64::MAX)];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}

#[test]
fn test_discrete() {
	let cases = [((5.0, 0.3), 35), ((10.0, 0.7), 21)];
	for (args, max_val) in cases {
		check_discrete_distribution(&new_dist(args), max_val);
	}
}

#[test]
#[cfg(feature = "rand")]
fn test_sample() {
	use rand::{SeedableRng, distr::Distribution, rngs::StdRng};

	let dist = NegativeBinomial::new(4.0, 0.5).unwrap();
	let mut rng = StdRng::seed_from_u64(1600);
	let n_samples = 10_000;
	let tol = 0.1;

	let samples: Vec<u64> =
		dist.sample_iter(&mut rng).take(n_samples).collect();
	let sample_mean = samples.iter().sum::<u64>() as f64 / n_samples as f64;
	let sample_variance = samples
		.iter()
		.map(|&x| (x as f64 - sample_mean).powi(2))
		.sum::<f64>() / n_samples as f64;

	let theoretical_mean = dist.mean().unwrap();
	let theoretical_variance = dist.variance().unwrap();

	assert_almost_eq!(sample_mean, theoretical_mean, epsilon = tol);
	assert_almost_eq!(sample_variance, theoretical_variance, epsilon = tol);
}
