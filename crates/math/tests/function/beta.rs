use core::f64::consts::*;
use math::assert_almost_eq;
use math::function::beta::*;

#[test]
fn test_ln_beta() {
	let cases = [
		((0.5, 0.5), 1.1447298858494002, 1e-15),
		((1.0, 0.5), LN_2, 1e-14),
		((2.5, 0.5), 0.16390063283767395, 1e-15),
		((0.5, 1.0), LN_2, 1e-14),
		((1.0, 1.0), 0.0, 1e-15),
		((2.5, 1.0), -0.9162907318741551, 1e-14),
		((0.5, 2.5), 0.16390063283767395, 1e-15),
		((1.0, 2.5), -0.9162907318741551, 1e-14),
		((2.5, 2.5), -2.6086880894021074, 1e-14),
	];

	for ((a, b), expected, epsilon) in cases {
		assert_almost_eq!(ln_beta(a, b), expected, epsilon)
	}
}

#[test]
#[should_panic]
fn test_ln_beta_a_lte_0() {
	ln_beta(0.0, 0.5);
}

#[test]
#[should_panic]
fn test_ln_beta_b_lte_0() {
	ln_beta(0.5, 0.0);
}

#[test]
fn test_checked_ln_beta_a_lte_0() {
	assert!(checked_ln_beta(0.0, 0.5).is_err());
}

#[test]
fn test_checked_ln_beta_b_lte_0() {
	assert!(checked_ln_beta(0.5, 0.0).is_err());
}

#[test]
#[should_panic]
fn test_beta_a_lte_0() {
	beta(0.0, 0.5);
}

#[test]
#[should_panic]
fn test_beta_b_lte_0() {
	beta(0.5, 0.0);
}

#[test]
fn test_checked_beta_a_lte_0() {
	assert!(checked_beta(0.0, 0.5).is_err());
}

#[test]
fn test_checked_beta_b_lte_0() {
	assert!(checked_beta(0.5, 0.0).is_err());
}

#[test]
fn test_beta() {
	let cases = [
		((0.5, 0.5), PI, 1e-15),
		((1.0, 0.5), 2.0, 1e-14),
		((2.5, 0.5), 1.1780972450961724, 1e-15),
		((0.5, 1.0), 2.0, 1e-14),
		((1.0, 1.0), 1.0, 1e-15),
		((2.5, 1.0), 0.4, 1e-14),
		((0.5, 2.5), 1.1780972450961724, 1e-15),
		((1.0, 2.5), 0.4, 1e-14),
		((2.5, 2.5), 0.07363107781851078, 1e-15),
	];

	for ((a, b), expected, epsilon) in cases {
		assert_almost_eq!(beta(a, b), expected, epsilon)
	}
}

#[test]
fn test_beta_inc() {
	let cases = [
		((0.5, 0.5, 0.5), FRAC_PI_2, 1e-14),
		((0.5, 0.5, 1.0), PI, 1e-15),
		((1.0, 0.5, 0.5), 0.585786437626905, 1e-15),
		((1.0, 0.5, 1.0), 2.0, 1e-14),
		((2.5, 0.5, 0.5), 0.08904862254808624, 1e-16),
		((2.5, 0.5, 1.0), 1.1780972450961724, 1e-15),
		((0.5, 1.0, 0.5), SQRT_2, 1e-14),
		((0.5, 1.0, 1.0), 2.0, 1e-14),
		((1.0, 1.0, 0.5), 0.5, 1e-15),
		((1.0, 1.0, 1.0), 1.0, 1e-15),
		((2.5, 1.0, 0.5), 0.07071067811865475, 1e-16),
		((2.5, 1.0, 1.0), 0.4, 1e-14),
		((0.5, 2.5, 0.5), 1.0890486225480862, 1e-15),
		((0.5, 2.5, 1.0), 1.1780972450961724, 1e-15),
		((1.0, 2.5, 0.5), 0.32928932188134524, 1e-14),
		((1.0, 2.5, 1.0), 0.4, 1e-14),
		((2.5, 2.5, 0.5), 0.03681553890925539, 1e-15),
		((2.5, 2.5, 1.0), 0.07363107781851078, 1e-15),
	];

	for ((a, b, x), expected, epsilon) in cases {
		assert_almost_eq!(beta_inc(a, b, x), expected, epsilon)
	}
}

#[test]
#[should_panic]
fn test_beta_inc_a_lte_0() {
	beta_inc(0.0, 1.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_inc_b_lte_0() {
	beta_inc(1.0, 0.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_inc_x_lt_0() {
	beta_inc(1.0, 1.0, -1.0);
}

#[test]
#[should_panic]
fn test_beta_inc_x_gt_1() {
	beta_inc(1.0, 1.0, 2.0);
}

#[test]
fn test_checked_beta_inc_a_lte_0() {
	assert!(checked_beta_inc(0.0, 1.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_inc_b_lte_0() {
	assert!(checked_beta_inc(1.0, 0.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_inc_x_lt_0() {
	assert!(checked_beta_inc(1.0, 1.0, -1.0).is_err());
}

#[test]
fn test_checked_beta_inc_x_gt_1() {
	assert!(checked_beta_inc(1.0, 1.0, 2.0).is_err());
}

#[test]
fn test_beta_reg() {
	let cases = [
		((0.5, 0.5, 0.5), 0.5, 1e-15),
		((0.5, 0.5, 1.0), 1.0, 1e-16),
		((1.0, 0.5, 0.5), 0.2928932188134525, 1e-15),
		((1.0, 0.5, 1.0), 1.0, 1e-16),
		((2.5, 0.5, 0.5), 0.07558681842161244, 1e-16),
		((2.5, 0.5, 1.0), 1.0, 1e-16),
		((0.5, 1.0, 0.5), FRAC_1_SQRT_2, 1e-15),
		((0.5, 1.0, 1.0), 1.0, 1e-16),
		((1.0, 1.0, 0.5), 0.5, 1e-15),
		((1.0, 1.0, 1.0), 1.0, 1e-16),
		((2.5, 1.0, 0.5), 0.1767766952966369, 1e-15),
		((2.5, 1.0, 1.0), 1.0, 1e-16),
		((0.5, 2.5, 0.5), 0.9244131815783876, 1e-16),
		((0.5, 2.5, 1.0), 1.0, 1e-16),
		((1.0, 2.5, 0.5), 0.8232233047033631, 1e-15),
		((1.0, 2.5, 1.0), 1.0, 1e-16),
		((2.5, 2.5, 0.5), 0.5, 1e-15),
		((2.5, 2.5, 1.0), 1.0, 1e-16),
	];

	for ((a, b, x), expected, epsilon) in cases {
		assert_almost_eq!(beta_reg(a, b, x), expected, epsilon)
	}
}

#[test]
#[should_panic]
fn test_beta_reg_a_lte_0() {
	beta_reg(0.0, 1.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_b_lte_0() {
	beta_reg(1.0, 0.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_x_lt_0() {
	beta_reg(1.0, 1.0, -1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_x_gt_1() {
	beta_reg(1.0, 1.0, 2.0);
}

#[test]
fn test_checked_beta_reg_a_lte_0() {
	assert!(checked_beta_reg(0.0, 1.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_reg_b_lte_0() {
	assert!(checked_beta_reg(1.0, 0.0, 1.0).is_err());
}

#[test]
fn test_checked_beta_reg_x_lt_0() {
	assert!(checked_beta_reg(1.0, 1.0, -1.0).is_err());
}

#[test]
fn test_checked_beta_reg_x_gt_1() {
	assert!(checked_beta_reg(1.0, 1.0, 2.0).is_err());
}

#[test]
fn test_error_is_sync_send() {
	fn assert_sync_send<T: Sync + Send>() {}
	assert_sync_send::<BetaFuncError>();
}
