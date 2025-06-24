use math::function::evaluate::*;

// TODO: more robust tests
#[test]
fn test_polynomial() {
	let empty: [f64; 0] = [];
	assert_eq!(polynomial(2.0, &empty), 0.0);

	let zero = [0.0];
	assert_eq!(polynomial(2.0, &zero), 0.0);

	let mut coeff = [1.0, 0.0, 5.0];
	assert_eq!(polynomial(2.0, &coeff), 21.0);

	coeff = [-5.0, -2.0, 3.0];
	assert_eq!(polynomial(2.0, &coeff), 3.0);
	assert_eq!(polynomial(-2.0, &coeff), 11.0);

	let large_coeff = [-1.35e3, 2.5e2, 8.0, -4.0, 1e2, 3.0];
	assert_eq!(polynomial(5.0, &large_coeff), 71475.0);
	assert_eq!(polynomial(-5.0, &large_coeff), 51225.0);

	coeff = [f64::INFINITY, -2.0, 3.0];
	assert_eq!(polynomial(2.0, &coeff), f64::INFINITY);
	assert_eq!(polynomial(-2.0, &coeff), f64::INFINITY);

	coeff = [f64::NEG_INFINITY, -2.0, 3.0];
	assert_eq!(polynomial(2.0, &coeff), f64::NEG_INFINITY);
	assert_eq!(polynomial(-2.0, &coeff), f64::NEG_INFINITY);

	coeff = [f64::NAN, -2.0, 3.0];
	assert!(polynomial(2.0, &coeff).is_nan());
	assert!(polynomial(-2.0, &coeff).is_nan());
}
