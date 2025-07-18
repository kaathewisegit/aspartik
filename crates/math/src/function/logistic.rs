//! Provides the [logistic](http://en.wikipedia.org/wiki/Logistic_function) and
//! related functions

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// [Logistic function][wiki]
///
/// [wiki]: https://en.wikipedia.org/wiki/Logistic_function
#[cfg_attr(feature = "python", pyfunction)]
pub fn logistic(p: f64) -> f64 {
	1.0 / ((-p).exp() + 1.0)
}

/// [Logit function][wiki]
///
/// Returns `None` if `p` is not in `[0, 1]`.
///
/// [wiki]: https://en.wikipedia.org/wiki/Logit
#[cfg_attr(feature = "python", pyfunction)]
pub fn logit(p: f64) -> Option<f64> {
	if (0.0..=1.0).contains(&p) {
		Some((p / (1.0 - p)).ln())
	} else {
		None
	}
}
