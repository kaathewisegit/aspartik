use indoc::indoc;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::fmt::Display;

/// Targeted accuracy instantiated over `f64`
pub const ACCURACY: f64 = 1e-10;

/// Standard epsilon, maximum relative precision of IEEE 754 double-precision
/// floating point numbers (64 bit) e.g. `2^-53`
pub const F64_PREC: f64 = 0.00000000000000011102230246251565;

pub const DEFAULT_F64_ACC: f64 = F64_PREC * 10.0;

#[macro_export]
macro_rules! assert_almost_eq {
	($a:expr, $b:expr, $epsilon:expr $(,)?) => {{
		use ::math::tolerance::{is_close, Tolerance};

		if !is_close($a, $b, $epsilon, 1e-16, 4) {
			panic!(
				"assert_almost_eq!({}, {}) failed:\n{}",
				stringify!($a),
				stringify!($b),
				::math::tolerance::report($a, $b),
			);
		}
	}};
	($a:expr, $b:expr $(,)?) => {
		assert_almost_eq!($a, $b, f64::EPSILON);
	};
}

/// Methods for calculating differences between numbers.
///
/// Used primarily for testing.
pub trait Tolerance {
	fn abs_diff(self, other: Self) -> Self;
	fn relative(self, other: Self) -> Self;

	/// Calculates the difference in [units in last place][ulps].
	///
	/// [ulps]: https://en.wikipedia.org/wiki/Unit_in_the_last_place
	fn ulps(self, other: Self) -> u64;
}

impl Tolerance for f64 {
	fn abs_diff(self, other: Self) -> Self {
		(self - other).abs()
	}

	// XXX: is this a good algorithm?
	fn relative(self, other: Self) -> Self {
		if self == other {
			return 0.0;
		}

		let a = self.abs();
		let b = other.abs();

		let (a, b) = (a.max(b), a.min(b));
		// `a` can't be zero, because both `a >= 0`, `b >= 0`, and if
		// `a` is a zero, then `b` must be too, but in this case the
		// function would've returned early in the first condition

		(a - b) / a
	}

	/// ULPS calculated with a simple bitcast.
	///
	/// This function will return nonsense values if either of the operands
	/// is infinite or NaN.
	fn ulps(self, other: Self) -> u64 {
		let self_bits = self.to_bits();
		let other_bits = other.to_bits();

		// XXX: what if one of them is zero and the other is close to
		// zero?

		self_bits.abs_diff(other_bits)
	}
}

pub fn report<T>(given: T, expected: T) -> String
where
	T: Display + Copy + Tolerance,
{
	format!(
		indoc!("
			   given: {}
			expected: {}
			 ---------
			abs diff: {}
			relative: {}
			    ulps: {}"),
		given,
		expected,
		given.abs_diff(expected),
		given.relative(expected),
		given.ulps(expected),
	)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (
	given, expected,
	epsilon = f64::EPSILON, relative = 1e-15, ulps = 4,
))]
pub fn is_close(
	given: f64,
	expected: f64,
	epsilon: f64,
	relative: f64,
	ulps: u64,
) -> bool {
	let abs_diff = given.abs_diff(expected);
	let largest = given.max(expected);

	abs_diff <= epsilon
		|| abs_diff <= largest * relative
		|| given.ulps(expected) <= ulps
}
