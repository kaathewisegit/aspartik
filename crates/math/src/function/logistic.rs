//! Provides the [logistic](http://en.wikipedia.org/wiki/Logistic_function) and
//! related functions

/// Computes the logistic function
pub fn logistic(p: f64) -> f64 {
	1.0 / ((-p).exp() + 1.0)
}

/// Computes the logit function
///
/// # Panics
///
/// If `p < 0.0` or `p > 1.0`
pub fn logit(p: f64) -> f64 {
	checked_logit(p).unwrap()
}

/// Computes the logit function, returning `None` if `p < 0.0` or `p > 1.0`.
pub fn checked_logit(p: f64) -> Option<f64> {
	if (0.0..=1.0).contains(&p) {
		Some((p / (1.0 - p)).ln())
	} else {
		None
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert_almost_eq;

	#[test]
	fn test_logistic_bounds() {
		assert_eq!(logistic(f64::NEG_INFINITY), 0.0);
		assert_eq!(logistic(f64::INFINITY), 1.0);
	}

	#[test]
	fn test_logistic() {
		let cases = [
			(-11.512915464920228103874353849992239636376994324587, 0.00001, 1e-16),
			(-6.9067547786485535272274487616830597875179908939086, 0.001, 1e-18),
			(-2.1972245773362193134015514347727700402304323440139, 0.1, 1e-16),
			(0.0, 0.5, 1e-16),
			(2.1972245773362195801634726294284168954491240598975, 0.9, 1e-15),
			(6.9067547786485526081487245019905638981131702804661, 0.999, 1e-15),
			(11.512915464924779098232747799811946290419057060965, 0.99999, 1e-16),
		];

		for (input, expected, epsilon) in cases {
			assert_almost_eq!(logistic(input), expected, epsilon);
		}
	}

	#[test]
	fn test_logit_bounds() {
		assert_eq!(logit(0.0), f64::NEG_INFINITY);
		assert_eq!(logit(1.0), f64::INFINITY);
	}

	#[test]
	fn test_logit() {
		let cases = [
			(0.00001, -11.512915464920228103874353849992239636376994324587, 1e-16),
			(0.001, -6.9067547786485535272274487616830597875179908939086, 1e-16),
			(0.1, -2.1972245773362193134015514347727700402304323440139, 1e-16),
			(0.5, 0.0, 1e-16),
			(0.9, 2.1972245773362195801634726294284168954491240598975, 1e-16),
			(0.999, 6.9067547786485526081487245019905638981131702804661, 1e-16),
			(0.99999, 11.512915464924779098232747799811946290419057060965, 1e-16),
		];

		for (input, expected, epsilon) in cases {
			assert_almost_eq!(logit(input), expected, epsilon);
		}
	}

	#[test]
	#[should_panic]
	fn test_logit_p_lt_0() {
		logit(-1.0);
	}

	#[test]
	#[should_panic]
	fn test_logit_p_gt_1() {
		logit(2.0);
	}

	#[test]
	fn test_checked_logit_p_lt_0() {
		assert!(checked_logit(-1.0).is_none());
	}

	#[test]
	fn test_checked_logit_p_gt_1() {
		assert!(checked_logit(2.0).is_none());
	}
}
