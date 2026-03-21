use stats::distribution::{Categorical, CategoricalError};

use crate::prelude::*;

make_test_harness!(Categorical(prob_mass: &[f64]), CategoricalError);

#[test]
fn test_new_is_ok() {
	let cases: &[&[f64]] = &[
		&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
		&[0.0, 0.25, 0.5, 0.25],
		&[0.0, 0.5, 0.5],
		&[0.75, 0.25],
		&[1.0, 0.0, 1.0],
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases: &[(&[f64], CategoricalError)] = &[
		(&[], CategoricalError::ProbMassEmpty),
		(&[-1.0, 1.0], CategoricalError::ProbMassHasInvalidElements),
		(&[0.0, 0.0, 0.0], CategoricalError::ProbMassSumZero),
	];
	for (args, err) in cases {
		assert_new_is_err(args, *err);
	}
}

#[test]
fn test_mean() {
	let cases: &[(&[f64], f64)] = &[
		(&[0.0, 0.25, 0.5, 0.25], 2.0),
		(&[0.0, 1.0, 2.0, 1.0], 2.0),
		(&[0.0, 0.5, 0.5], 1.5),
		(&[0.75, 0.25], 0.25),
		(
			&[
				1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
				1.0, 1.0,
			],
			5.0,
		),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), *expected);
	}
}

#[test]
fn test_variance() {
	let cases: &[(&[f64], f64)] = &[
		(&[0.0, 0.25, 0.5, 0.25], 0.5),
		(&[0.0, 1.0, 2.0, 1.0], 0.5),
		(&[0.0, 0.5, 0.5], 0.25),
		(&[0.75, 0.25], 0.1875),
		(&[1.0, 0.0, 1.0], 1.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.variance().unwrap(), *expected);
	}
}

#[test]
fn test_entropy() {
	let cases: &[(&[f64], f64)] = &[
		(&[0.0, 1.0], 0.0),
		(&[0.0, 1.0, 1.0], 2f64.ln()),
		(&[1.0, 1.0, 1.0], 3f64.ln()),
		(&vec![1.0; 100], 100f64.ln()),
		(&[0.0, 0.25, 0.5, 0.25], 1.0397207708399179),
	];
	for (args, expected) in cases {
		assert_close(args, (), |d, _| d.entropy().unwrap(), *expected);
	}
}

#[test]
fn test_median() {
	let cases: &[(&[f64], f64)] =
		&[(&[0.0, 3.0, 1.0, 1.0], 1.0), (&[4.0, 2.5, 2.5, 1.0], 1.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.median().unwrap(), *expected);
	}
}

#[test]
fn test_lower() {
	let cases: &[(&[f64], u64)] = &[(&[4.0, 2.5, 2.5, 1.0], 0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), *expected);
	}
}

#[test]
fn test_upper() {
	let cases: &[(&[f64], u64)] = &[(&[4.0, 2.5, 2.5, 1.0], 3)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), *expected);
	}
}

#[test]
fn test_pmf() {
	let cases: &[(&[f64], u64, f64)] = &[
		(&[0.0, 0.25, 0.5, 0.25], 0, 0.0),
		(&[0.0, 0.25, 0.5, 0.25], 1, 0.25),
		(&[0.0, 0.25, 0.5, 0.25], 3, 0.25),
		(&[4.0, 2.5, 2.5, 1.0], 4, 0.0),
	];
	for (args, x, expected) in cases {
		assert_exact(args, x, |d, x| d.pmf(*x), *expected);
	}
}

#[test]
fn test_ln_pmf() {
	let cases: &[(&[f64], u64, f64)] = &[
		(&[0.0, 0.25, 0.5, 0.25], 0, 0f64.ln()),
		(&[0.0, 0.25, 0.5, 0.25], 1, 0.25f64.ln()),
		(&[0.0, 0.25, 0.5, 0.25], 3, 0.25f64.ln()),
		(&[4.0, 2.5, 2.5, 1.0], 4, f64::NEG_INFINITY),
	];
	for (args, x, expected) in cases {
		assert_exact(args, x, |d, x| d.ln_pmf(*x), *expected);
	}
}

#[test]
fn test_cdf() {
	let cases: &[(&[f64], u64, f64)] = &[
		(&[0.0, 3.0, 1.0, 1.0], 1, 3.0 / 5.0),
		(&[1.0, 1.0, 1.0, 1.0], 0, 0.25),
		(&[4.0, 2.5, 2.5, 1.0], 0, 0.4),
		(&[4.0, 2.5, 2.5, 1.0], 3, 1.0),
		(&[4.0, 2.5, 2.5, 1.0], 4, 1.0),
	];
	for (args, x, expected) in cases {
		assert_exact(args, x, |d, x| d.cdf(*x), *expected);
	}
}

#[test]
fn test_sf() {
	let cases: &[(&[f64], u64, f64)] = &[
		(&[0.0, 3.0, 1.0, 1.0], 1, 2.0 / 5.0),
		(&[1.0, 1.0, 1.0, 1.0], 0, 0.75),
		(&[4.0, 2.5, 2.5, 1.0], 0, 0.6),
		(&[4.0, 2.5, 2.5, 1.0], 3, 0.0),
		(&[4.0, 2.5, 2.5, 1.0], 4, 0.0),
	];
	for (args, x, expected) in cases {
		assert_exact(args, x, |d, x| d.sf(*x), *expected);
	}
}

#[test]
fn test_cdf_sf_mirror() {
	let mass = [4.0, 2.5, 2.5, 1.0];
	for x in [0, 1, 2, 3] {
		assert_close(&mass, x, |d, x| d.cdf(x) + d.sf(x), 1.0);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases: &[(&[f64], f64, u64)] = &[
		(&[0.0, 3.0, 1.0, 1.0], 0.2, 1),
		(&[0.0, 3.0, 1.0, 1.0], 0.5, 1),
		(&[0.0, 3.0, 1.0, 1.0], 0.95, 3),
		(&[4.0, 2.5, 2.5, 1.0], 0.2, 0),
		(&[4.0, 2.5, 2.5, 1.0], 0.5, 1),
		(&[4.0, 2.5, 2.5, 1.0], 0.95, 3),
	];
	for (args, p, expected) in cases {
		let p = Probability::new(*p);
		assert_exact(args, p, |d, p| d.inverse_cdf(p), *expected);
	}
}

#[test]
fn test_discrete() {
	check_discrete_distribution(&new_dist(&[1.0, 2.0, 3.0, 4.0]), 4);
	check_discrete_distribution(&new_dist(&[0.0, 1.0, 2.0, 3.0, 4.0]), 5);
}
