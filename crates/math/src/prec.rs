//! Utility functions for working with floating point precision

/// Targeted accuracy instantiated over `f64`
pub const ACCURACY: f64 = 1e-10;

/// Standard epsilon, maximum relative precision of IEEE 754 double-precision
/// floating point numbers (64 bit) e.g. `2^-53`
pub const F64_PREC: f64 = 0.00000000000000011102230246251565;

/// Default accuracy for `f64`, equivalent to `10.0 * F64_PREC`
pub const DEFAULT_F64_ACC: f64 = 0.0000000000000011102230246251565;

/// A shorthand version of `approx::assert_abs_diff_eq`.
#[macro_export]
macro_rules! assert_almost_eq {
	($a:expr, $b:expr, $epsilon:expr $(,)?) => {
		::approx::assert_abs_diff_eq!($a, $b, epsilon = $epsilon);
	};
	($a:expr, $b:expr, $(,)?) => {
		::approx::assert_abs_diff_eq!($a, $b);
	};
}

/// Compares if two floats are close via `approx::relative_eq!`
/// and `crate::consts::ACC` relative precision.
/// Updates first argument to value of second argument
pub fn convergence(x: &mut f64, x_new: f64) -> bool {
	let res = approx::relative_eq!(*x, x_new, max_relative = ACCURACY);
	*x = x_new;
	res
}
