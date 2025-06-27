use indoc::indoc;

use std::fmt::Display;

/// Targeted accuracy instantiated over `f64`
pub const ACCURACY: f64 = 1e-10;

/// Standard epsilon, maximum relative precision of IEEE 754 double-precision
/// floating point numbers (64 bit) e.g. `2^-53`
pub const F64_PREC: f64 = 0.00000000000000011102230246251565;

pub const DEFAULT_F64_ACC: f64 = F64_PREC * 10.0;

/// A shorthand version of `approx::assert_abs_diff_eq`.
#[macro_export]
macro_rules! assert_almost_eq {
	($a:expr, $b:expr, $epsilon:expr $(,)?) => {
		if !::approx::relative_eq!($a, $b, epsilon = $epsilon) {
			panic!(
				"assert_almost_eq!({}, {}) failed:\n{}",
				stringify!($a),
				stringify!($b),
				::math::tolerance::report($a, $b),
			);
		}
	};
	($a:expr, $b:expr $(,)?) => {
		if !::approx::abs_diff_eq!($a, $b) {
			panic!(
				"assert_almost_eq!({}, {}) failed:\n{}",
				stringify!($a),
				stringify!($b),
				::math::tolerance::report($a, $b),
			);
		}
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
		if self > other {
			self / other - 1.0
		} else {
			other / self - 1.0
		}
	}

	/// ULPS calculated with a simple bitcast.
	///
	/// This function will return nonsense values if either of the operands
	/// is infinite or NaN.
	fn ulps(self, other: Self) -> u64 {
		let self_bits = self.to_bits();
		let other_bits = other.to_bits();

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
