use stats::distribution::Chi;

use crate::prelude::*;
use std::{f64::consts::SQRT_2, num::NonZeroU64};

make_test_harness!(Chi(freedom: NonZeroU64));

fn nz(value: u64) -> NonZeroU64 {
	NonZeroU64::new(value).unwrap()
}

#[test]
fn test_mean() {
	let cases = [
		(1, 0.7978845608028654),
		(2, 1.2533141373155003),
		(5, 2.127692162140974),
		(336, 18.31666925443713),
	];
	for (args, expected) in cases {
		assert_close(nz(args), (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_large_dof_mean_not_nan() {
	for i in 1..1000 {
		let mean = Chi::new(nz(i)).mean().unwrap();
		assert!(!mean.is_nan(), "Chi mean for {i} dof was {mean}");
	}
}

#[test]
fn test_variance() {
	let cases = [
		(1, 0.3633802276324187),
		(2, 0.4292036732051034),
		(3, 0.45352091052967464),
	];
	for (args, expected) in cases {
		assert_close(
			nz(args),
			(),
			|d, _| d.variance().unwrap(),
			expected,
		);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		(1, 0.7257913526447274),
		(2, 0.9420342421707938),
		(3, 0.9961541981062056),
	];
	for (args, expected) in cases {
		assert_close(
			nz(args),
			(),
			|d, _| d.entropy().unwrap(),
			expected,
		);
	}
}

#[test]
fn test_skewness() {
	let cases = [(1, 0.995271746431156), (3, 0.4856928280495908)];
	for (args, expected) in cases {
		assert_close(
			nz(args),
			(),
			|d, _| d.skewness().unwrap(),
			expected,
		);
	}
}

#[test]
fn test_mode() {
	let cases = [(1, 0.0), (2, 1.0), (3, SQRT_2)];
	for (args, expected) in cases {
		assert_close(nz(args), (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [(1, 0.0), (2, 0.0), (3, 0.0)];
	for (args, expected) in cases {
		assert_close(nz(args), (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases =
		[(1, f64::INFINITY), (2, f64::INFINITY), (3, f64::INFINITY)];
	for (args, expected) in cases {
		assert_close(nz(args), (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		(1, 0.0, 0.0),
		(1, 0.1, 0.7939050949540235),
		(1, 1.0, 0.4839414490382867),
		(1, 5.5, 2.1539520085086552e-7),
		(1, f64::INFINITY, 0.0),
		(2, 0.0, 0.0),
		(2, 0.1, 0.09950124791926823),
		(2, 1.0, 0.6065306597126334),
		(2, 5.5, 1.4847681768496578e-6),
		(2, f64::INFINITY, 0.0),
		(170, 13.0, 0.5644678498668441),
	];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_neg_pdf() {
	let cases = [(1, -1.0, 0.0)];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		(1, 0.0, f64::NEG_INFINITY),
		(1, 0.1, -0.23079135264472744),
		(1, 1.0, -0.7257913526447274),
		(1, 5.5, -15.350791352644727),
		(1, f64::INFINITY, f64::NEG_INFINITY),
		(2, 0.0, f64::NEG_INFINITY),
		(2, 0.1, -2.307585092994046),
		(2, 1.0, -0.5),
		(2, 5.5, -13.420251907761575),
		(2, f64::INFINITY, f64::NEG_INFINITY),
		(170, 13.0, -0.5718718503060052),
	];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_neg_ln_pdf() {
	let cases = [(1, -1.0, f64::NEG_INFINITY)];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		(1, 0.0, 0.0),
		(1, 0.1, 0.07965567455405796),
		(1, 1.0, 0.6826894921370859),
		(1, 5.5, 0.999999962020875),
		(1, f64::INFINITY, 1.0),
		(2, 0.0, 0.0),
		(2, 0.1, 0.004987520807317686),
		(2, f64::INFINITY, 1.0),
	];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		(1, 0.0, 1.0),
		(1, 0.1, 0.920344325445942),
		(1, 1.0, 0.31731050786291404),
		(1, 5.5, 3.797912493177544e-8),
		(1, f64::INFINITY, 0.0),
		(2, 0.0, 1.0),
		(2, 0.1, 0.9950124791926823),
		(2, 1.0, 0.6065306597126333),
		(2, 5.5, 2.699578503363014e-7),
		(2, f64::INFINITY, 0.0),
	];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_neg_cdf() {
	let cases = [(1, -1.0, 0.0)];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_neg_sf() {
	let cases = [(1, -1.0, 1.0)];
	for (args, p, expected) in cases {
		assert_close(nz(args), p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_continuous() {
	for i in [1, 2, 5] {
		check_continuous_distribution(&Chi::new(nz(i)), 0.0, 10.0);
	}
}
