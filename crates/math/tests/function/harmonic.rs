use math::{assert_almost_eq, function::harmonic::*};

#[test]
fn test_harmonic() {
	assert_eq!(harmonic(0), 1.0);

	let cases = [
		(1, 1.0, 1e-14),
		(2, 1.5, 1e-14),
		(4, 2.0833333333333335, 1e-14),
		(8, 2.717857142857143, 1e-14),
		(16, 3.3807289932289932, 1e-14),
	];
	for (x, expected, epsilon) in cases {
		assert_almost_eq!(harmonic(x), expected, epsilon);
	}
}

#[test]
fn test_gen_harmonic_exact() {
	let cases = [
		((0, 0.0), 1.0),
		((0, f64::INFINITY), 1.0),
		((0, f64::NEG_INFINITY), 1.0),
		((1, 0.0), 1.0),
		((1, f64::INFINITY), 1.0),
		((1, f64::NEG_INFINITY), 1.0),
		((2, 1.0), 1.5),
		((2, 3.0), 1.125),
		((2, f64::INFINITY), 1.0),
		((2, f64::NEG_INFINITY), f64::INFINITY),
		((4, f64::INFINITY), 1.0),
		((4, f64::NEG_INFINITY), f64::INFINITY),
	];

	for ((n, m), expected) in cases {
		assert_eq!(gen_harmonic(n, m), expected);
	}
}

#[test]
fn test_gen_harmonic() {
	let cases = [
		((4, 1.0), 2.0833333333333335, 1e-14),
		((4, 3.0), 1.177662037037037, 1e-16),
	];

	for ((n, m), expected, epsilon) in cases {
		assert_almost_eq!(gen_harmonic(n, m), expected, epsilon);
	}
}
