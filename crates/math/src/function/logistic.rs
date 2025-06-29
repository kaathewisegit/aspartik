//! Provides the [logistic](http://en.wikipedia.org/wiki/Logistic_function) and
//! related functions

/// Computes the logistic function
pub fn logistic(p: f64) -> f64 {
	1.0 / ((-p).exp() + 1.0)
}

/// Computes the logit function, returning `None` if `p < 0.0` or `p > 1.0`.
pub fn logit(p: f64) -> Option<f64> {
	if (0.0..=1.0).contains(&p) {
		Some((p / (1.0 - p)).ln())
	} else {
		None
	}
}
