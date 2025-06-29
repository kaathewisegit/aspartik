use math::{assert_almost_eq, function::logistic::*};

#[test]
fn test_logistic_bounds() {
	assert_eq!(logistic(f64::NEG_INFINITY), 0.0);
	assert_eq!(logistic(f64::INFINITY), 1.0);
}

#[test]
fn test_logistic() {
	let cases = [
		(-11.512915464920228, 0.00001, 1e-16),
		(-6.906754778648554, 0.001, 1e-18),
		(-2.197224577336219, 0.1, 1e-16),
		(0.0, 0.5, 1e-16),
		(2.1972245773362196, 0.9, 1e-15),
		(6.906754778648553, 0.999, 1e-15),
		(11.51291546492478, 0.99999, 1e-16),
	];

	for (input, expected, epsilon) in cases {
		assert_almost_eq!(logistic(input), expected, epsilon);
	}
}

#[test]
fn test_logit_bounds() {
	assert_eq!(logit(0.0).unwrap(), f64::NEG_INFINITY);
	assert_eq!(logit(1.0).unwrap(), f64::INFINITY);
}

#[test]
fn test_logit() {
	let cases = [
		(0.00001, -11.512915464920228, 1e-16),
		(0.001, -6.906754778648554, 1e-16),
		(0.1, -2.197224577336219, 1e-16),
		(0.5, 0.0, 1e-16),
		(0.9, 2.1972245773362196, 1e-16),
		(0.999, 6.906754778648553, 1e-16),
		(0.99999, 11.51291546492478, 1e-16),
	];

	for (input, expected, epsilon) in cases {
		assert_almost_eq!(logit(input).unwrap(), expected, epsilon);
	}
}

#[test]
fn test_logit_p_lt_0() {
	assert!(logit(-1.0).is_none());
}

#[test]
fn test_logit_p_gt_1() {
	assert!(logit(2.0).is_none());
}
