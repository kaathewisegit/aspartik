use stats::distribution::{Levy, LevyError};

use crate::prelude::*;

make_test_harness!(Levy(mu: f64, c: f64), LevyError);

#[test]
fn test_new_is_ok() {
	let cases = [(10.0, 0.1), (5.0, 1.0), (0.1, 10.0), (10.0, 100.0)];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((1.0, -1.0), LevyError::ScaleInvalid),
		((f64::NAN, 1.0), LevyError::LocationInvalid),
		((0.0, 0.0), LevyError::ScaleInvalid),
		((-1.0, -1.0), LevyError::ScaleInvalid),
		((1.0, f64::NAN), LevyError::ScaleInvalid),
		((f64::NAN, f64::NAN), LevyError::LocationInvalid),
		((f64::INFINITY, 1.0), LevyError::LocationInvalid),
		((1.0, f64::INFINITY), LevyError::ScaleInvalid),
		((f64::INFINITY, f64::INFINITY), LevyError::LocationInvalid),
		((f64::NEG_INFINITY, 1.0), LevyError::LocationInvalid),
		((1.0, f64::NEG_INFINITY), LevyError::ScaleInvalid),
		(
			(f64::NEG_INFINITY, f64::NEG_INFINITY),
			LevyError::LocationInvalid,
		),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [((1.0, 3.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [((1.0, 3.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		((0.1, 0.1), 1.0218977084028444),
		((1.0, 1.0), 3.32448280139689),
		((10.0, 10.0), 5.627067894390936),
		((3.0, 1.0), 3.32448280139689),
		((1.0, 3.0), 4.423095090065),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [
		((1.0, 1.0), 3.198109338317732),
		((1.0, 3.0), 7.594328014953197),
		((3.0, 1.0), 5.198109338317732),
		((3.0, 3.0), 9.594328014953197),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [
		((1.0, 1.0), 4.0 / 3.0),
		((1.0, 3.0), 2.0),
		((3.0, 1.0), 10.0 / 3.0),
		((3.0, 3.0), 4.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [
		((1.0, 1.0), 1.0),
		((1.0, 3.0), 1.0),
		((3.0, 1.0), 3.0),
		((3.0, 3.0), 3.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [
		((1.0, 1.0), f64::INFINITY),
		((1.0, 3.0), f64::INFINITY),
		((3.0, 1.0), f64::INFINITY),
		((3.0, 3.0), f64::INFINITY),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((1.0, 1.0), 1.0, 0.0),  // outside support
		((1.0, 1.0), -1.0, 0.0), // outside support
		((1.0, 1.0), 127.721, 0.0002785631041875089),
		((1.0, 3.0), 127.721, 0.0004786929706869084),
		((3.0, 1.0), 127.721, 0.00028527231427705446),
		((3.0, 3.0), 127.721, 0.0004901602905395954),
		((1.0, 10.0), 127.721, 0.0008501612885723883),
		((10.0, 1.0), 127.721, 0.00031101730916611916),
		((10.0, 10.0), 127.721, 0.0009466364648355231),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pdf(p), expected);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((1.0, 1.0), 1.0, f64::NEG_INFINITY), // outside support
		((1.0, 1.0), -1.0, f64::NEG_INFINITY), // outside support
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pdf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((1.0, 1.0), 1.0, 0.0),  // outside support
		((1.0, 1.0), -1.0, 0.0), // outside support
		((1.0, 1.0), 127.721, 0.9292144075830775),
		((1.0, 3.0), 127.721, 0.8777171617689964),
		((3.0, 1.0), 127.721, 0.9286506165184578),
		((3.0, 3.0), 127.721, 0.8767483839121537),
		((1.0, 10.0), 127.721, 0.7787752111524361),
		((10.0, 1.0), 127.721, 0.9265657651297626),
		((10.0, 10.0), 127.721, 0.7707025761090431),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((1.0, 1.0), 0.99, 6366.864385106222),
		((1.0, 3.0), 0.99, 19098.593155318664),
		((3.0, 1.0), 0.99, 6368.864385106222),
		((3.0, 3.0), 0.99, 19100.593155318664),
		((1.0, 10.0), 0.99, 63659.64385106222),
		((10.0, 1.0), 0.99, 6375.864385106222),
		((10.0, 10.0), 0.99, 63668.64385106222),
	];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.inverse_cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((1.0, 1.0), 1.0, 1.0),  // outside support
		((1.0, 1.0), -1.0, 1.0), // outside support
		((1.0, 1.0), 127.721, 0.07078559241692249),
		((1.0, 3.0), 127.721, 0.1222828382310036),
		((3.0, 1.0), 127.721, 0.07134938348154218),
		((3.0, 3.0), 127.721, 0.12325161608784627),
		((1.0, 10.0), 127.721, 0.2212247888475639),
		((10.0, 1.0), 127.721, 0.07343423487023741),
		((10.0, 10.0), 127.721, 0.22929742389095684),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_continuous() {
	let cases = [
		((1.0, 0.05), 1.0, 319.2932192553111),
		((3.0, 0.05), 3.0, 321.2932192553111),
		((10.0, 0.05), 10.0, 328.2932192553111),
	];
	for (args, lower, upper) in cases {
		check_continuous_distribution(&new_dist(args), lower, upper);
	}
}
