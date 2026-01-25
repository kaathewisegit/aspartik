#[macro_export]
macro_rules! py_bail {
	($type:ident, $($arg:tt)*) => {
		return Err($type::new_err(format!($($arg)*)).into());
	}
}

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

#[macro_export]
macro_rules! py_pickle_state_impl {
	($class:ident, $mod:ident) => {
		mod $mod {
			use super::$class;

			use anyhow::Result;
			use parking_lot::Mutex;
			use pyo3::prelude::*;
			use pyo3::type_object::PyTypeInfo;
			use pyo3::types::{PyBytes, PyType};

			#[pymethods]
			impl $class {
				#[classmethod]
				fn deserialize(
					_cls: Py<PyType>,
					bytes: &[u8],
				) -> Result<Self> {
					let inner =
						rmp_serde::from_slice(bytes)?;
					Ok(Self {
						inner: Mutex::new(inner),
					})
				}

				fn __reduce__(
					&self,
					py: Python,
				) -> Result<(Py<PyAny>, Py<PyAny>)> {
					let inner = &*self.inner.lock();
					let vec = rmp_serde::to_vec(inner)?;

					let pytype = Self::type_object(py);
					let method =
						pytype.getattr("deserialize")?;

					Ok((
						method.into(),
						(vec,).into_pyobject(py)?
							.into(),
					))
				}
			}
		}
	};
}

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

#[macro_export]
macro_rules! py_get_attr {
	($obj:expr, $name:literal) => {
		$obj.getattr(::pyo3::intern!($obj.py(), $name))
	};
}

#[macro_export]
macro_rules! py_extract_attr {
	($obj:expr, $name:literal, $type:ty $(,)?) => {
		$crate::py_get_attr!($obj, $name)
			.and_then(|attr| attr.extract::<$type>())
	};
}

#[macro_export]
macro_rules! py_has_method {
	($obj:expr, $name:expr) => {{
		let method = $crate::py_get_attr!($obj, $name);
		method.is_ok_and(|m| m.is_callable())
	}};
}

#[macro_export]
macro_rules! py_check_method {
	($obj:expr, $name:expr) => {{
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
