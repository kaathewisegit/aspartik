use math::{assert_almost_eq, function::factorial::*};

#[test]
fn test_factorial_and_ln_factorial() {
	let mut fac = 1.0;
	assert_eq!(factorial(0), fac);
	for i in 1..171 {
		fac *= i as f64;
		assert_eq!(factorial(i), fac);
		assert_eq!(ln_factorial(i), fac.ln());
	}
}

#[test]
fn test_factorial_overflow() {
	assert_eq!(factorial(172), f64::INFINITY);
	assert_eq!(factorial(u64::MAX), f64::INFINITY);
}

#[test]
fn test_ln_factorial_does_not_overflow() {
	assert_eq!(ln_factorial(1 << 10), 6078.211884750051);
	assert_almost_eq!(ln_factorial(1 << 12), 29978.648060844047, 1e-11);
	assert_eq!(ln_factorial(1 << 15), 307933.81973375485);
	assert_eq!(ln_factorial(1 << 17), 1413421.9939462072);
}

#[test]
fn test_binomial() {
	assert_eq!(binomial(1, 1), 1.0);
	assert_eq!(binomial(5, 2), 10.0);
	assert_eq!(binomial(7, 3), 35.0);
	assert_eq!(binomial(1, 0), 1.0);
	assert_eq!(binomial(0, 1), 0.0);
	assert_eq!(binomial(5, 7), 0.0);
}

#[test]
fn test_ln_binomial() {
	assert_eq!(ln_binomial(1, 1), 1f64.ln());
	assert_almost_eq!(ln_binomial(5, 2), 10f64.ln(), 1e-14);
	assert_almost_eq!(ln_binomial(7, 3), 35f64.ln(), 1e-14);
	assert_eq!(ln_binomial(1, 0), 1f64.ln());
	assert_eq!(ln_binomial(0, 1), 0f64.ln());
	assert_eq!(ln_binomial(5, 7), 0f64.ln());
}

#[test]
fn test_multinomial() {
	assert_eq!(1.0, multinomial(1, &[1, 0]));
	assert_eq!(10.0, multinomial(5, &[3, 2]));
	assert_eq!(10.0, multinomial(5, &[2, 3]));
	assert_eq!(35.0, multinomial(7, &[3, 4]));
}

#[test]
#[should_panic]
fn test_multinomial_bad_ni() {
	multinomial(1, &[1, 1]);
}

#[test]
fn test_checked_multinomial_bad_ni() {
	assert!(checked_multinomial(1, &[1, 1]).is_none());
}
