//! Cross-crate utils
//!
//! Currently they mostly contain PyO3 helpers.  Perhaps it would make sense to
//! upstream some of them.

pub mod atomic;

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns a Python exception of `$type` with a given message
///
/// The rest args have the same syntax as [`format_args!`].
#[macro_export]
macro_rules! py_bail {
	($type:ident, $($arg:tt)*) => {
		return Err($type::new_err(format!($($arg)*)).into());
	}
}

/// Calls a method named `$name` on the object `$obj`
///
/// `$name` is interned.  If there are rest parameters, they are passed as
/// positional arguments to [`call_method1`].  If there are dict `=>`
/// parameters, they are inserted into a `PyDict` which is passed to
/// [`call_method`].
///
/// [`call_method1`]: https://docs.rs/pyo3/latest/pyo3/struct.Py.html#method.call_method1
/// [`call_method`]: https://docs.rs/pyo3/latest/pyo3/struct.Py.html#method.call_method
#[macro_export]
macro_rules! py_call_method {
	($py:ident, $obj:expr, $name:literal) => {{
		use pyo3::intern;
		$obj.call_method0($py, intern!($py, $name))
	}};
	($py:ident, $obj:expr, $name:literal, $($arg:expr),+ $(,)?) => {{
		use pyo3::intern;
		$obj.call_method1($py, intern!($py, $name), ($($arg,)+))
	}};
	(
		$py:ident, $obj:expr, $name:literal,
		$($arg:expr,)* /,
		$($key:expr => $value:expr),+
		$(,)?
	) => {{
		use ::pyo3::{intern, types::PyDict};
		let kwargs = PyDict::new($py);
		$(
			kwargs.set_item($key, $value)?;
		)+
		$obj.call_method(
			$py,
			intern!($py, $name),
			($($arg,)*),
			Some(&kwargs)
		)
	}};
}

/// Implements `From<$err> for PyErr`
#[macro_export]
macro_rules! impl_pyerr {
	($err: ty, $pyexc: ty) => {
		impl std::convert::From<$err> for PyErr {
			fn from(err: $err) -> PyErr {
				<$pyexc>::new_err(err)
			}
		}
	};
}

/// Aspartik-specific utility which adds Rust submodules to `sys.modules`
#[macro_export]
macro_rules! py_patch_module {
	($m:ident) => {
		// https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
		::pyo3::py_run!(
			$m.py(),
			$m,
			&format!(
				"import sys; sys.modules['aspartik._aspartik_rust_impl.{}'] = m",
				$m.name()?,
			)
		);
	}
}

/// A convenience wrapper around `getattr` which interns `$name`
#[macro_export]
macro_rules! py_get_attr {
	($obj:expr, $name:literal) => {
		$obj.getattr(::pyo3::intern!($obj.py(), $name))
	};
}

/// Calls `py_get_attr` and then extracts the result into `$type`
#[macro_export]
macro_rules! py_extract_attr {
	($obj:expr, $name:literal, $type:ty $(,)?) => {
		$crate::py_get_attr!($obj, $name)
			.and_then(|attr| attr.extract::<$type>())
	};
}

/// Returns true if `$obj` has a method named `$name`
#[macro_export]
macro_rules! py_has_method {
	($obj:expr, $name:literal) => {{
		let method = $crate::py_get_attr!($obj, $name);
		method.is_ok_and(|m| m.is_callable())
	}};
}

/// Returns an error if `$obj` doesn't have a method named `$name`
#[macro_export]
macro_rules! py_check_method {
	($obj:expr, $name:literal) => {{
		use ::pyo3::exceptions::PyTypeError;

		if !$crate::py_has_method!($obj, $name) {
			$crate::py_bail!(
				PyTypeError,
				"method {} not found in {}",
				$name,
				$obj.get_type().name()?,
			);
		}
	}};
}

/// Times an operation
///
/// This macro executes `$e` and returns an `(out, time)` tuple, where `out` is
/// the output of the `$e` expression and `time` is the [`Duration`][d] of
/// `$e`'s execution.
///
/// [d]: std::time::Duration
#[macro_export]
macro_rules! time {
	($e:expr) => {{
		use ::std::time::Instant;
		let __start = Instant::now();
		let __out = $e;
		let __time = Instant::now() - __start;

		(__out, __time)
	}};
}

pub fn seconds_since_unix() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("time before the Unix epoch")
		.as_secs()
}
