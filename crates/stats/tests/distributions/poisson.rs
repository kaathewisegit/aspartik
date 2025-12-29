use stats::distribution::{Poisson, PoissonError};

use crate::prelude::*;

make_test_harness!(Poisson(lambda: f64), PoissonError);

#[test]
fn test_new_is_ok() {
	let cases = [1.5, 5.4, 10.8];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		(f64::NAN, PoissonError::LambdaInvalid),
		(-1.5, PoissonError::LambdaInvalid),
		(0.0, PoissonError::LambdaInvalid),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [(1.5, 1.5), (5.4, 5.4), (10.8, 10.8)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_variance() {
	let cases = [(1.5, 1.5), (5.4, 5.4), (10.8, 10.8)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.variance().unwrap(), expected);
	}
}

#[test]
fn test_entropy() {
	let cases = [
		(1.5, 1.5319591531023764),
		(5.4, 2.2449418395776437),
		(10.8, 2.6005964296769752),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), expected);
	}
}

#[test]
fn test_skewness() {
	let cases = [
		(1.5, 0.816496580927726),
		(5.4, 0.43033148291193524),
		(10.8, 0.3042903097250923),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.skewness().unwrap(), expected);
	}
}

#[test]
fn test_median() {
	let cases = [(1.5, 1.0), (5.4, 5.0), (10.8, 11.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), expected);
	}
}

#[test]
fn test_mode() {
	let cases = [(1.5, 1), (5.4, 5), (10.8, 10)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_lower() {
	let cases = [(1.5, 0), (5.4, 0), (10.8, 0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [(1.5, u64::MAX), (5.4, u64::MAX), (10.8, u64::MAX)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pmf() {
	let cases = [
		(1.5, 1, 0.334695240222645),
		(1.5, 10, 0.00000354574774057018),
		(1.5, 20, 0.000000000000000304971208961018),
		(5.4, 1, 0.0243895370901084),
		(5.4, 10, 0.0262412405917923),
		(5.4, 20, 0.000000825202200316548),
		(10.8, 1, 0.000220314636840657),
		(10.8, 10, 0.12136518365942),
		(10.8, 20, 0.00390813977857411),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.pmf(p), expected);
	}
}

#[test]
fn test_ln_pmf() {
	let cases = [
		(1.5, 1, -1.094534891891835),
		(1.5, 10, -12.549761491993873),
		(1.5, 20, -35.7263142985901),
		(5.4, 1, -3.7136010464297717),
		(5.4, 10, -3.640423037373228),
		(5.4, 20, -14.00763738934891),
		(10.8, 1, -8.420453865869826),
		(10.8, 10, -2.108951231773781),
		(10.8, 20, -5.544693778150009),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.ln_pmf(p), expected);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		(1.5, 1, 0.557825400371075),
		(1.5, 10, 0.999999448246764),
		(1.5, 20, 1.0),
		(5.4, 1, 0.0289061180327211),
		(5.4, 10, 0.977486300689765),
		(5.4, 20, 0.999999719992829),
		(10.8, 1, 0.000240714140251829),
		(10.8, 10, 0.483969235995569),
		(10.8, 20, 0.996180076960809),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		(1.5, 1, 0.44217459962892536),
		(1.5, 10, 0.0000005517532358246565),
		(1.5, 20, 2.3372210700347092e-17),
		(5.4, 1, 0.971093881967279),
		(5.4, 10, 0.022513699310235582),
		(5.4, 20, 0.0000002800071708975261),
		(10.8, 1, 0.9997592858597482),
		(10.8, 10, 0.5160307640044303),
		(10.8, 20, 0.003819923039191422),
	];
	for (args, p, expected) in cases {
		assert_close(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_discrete() {
	let cases = [(0.3, 10), (4.5, 30)];
	for (args, limit) in cases {
		check_discrete_distribution(&new_dist(args), limit);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [(1.5, 0.0, 0)];
	for (args, p, expected) in cases {
		assert_exact(
			args,
			p,
			|d, p| d.inverse_cdf(Probability::new(p).unwrap()),
			expected,
		);
	}
}
