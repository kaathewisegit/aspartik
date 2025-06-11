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
		use pyo3::{intern, types::PyDict};
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
			use bincode::{
				config::Configuration,
				serde::{decode_from_slice, encode_to_vec},
			};
			use pyo3::prelude::*;
			use pyo3::types::PyBytes;

			pub const BINCODE_CONFIG: Configuration =
				bincode::config::standard();

			#[pymethods]
			impl $class {
				fn __getstate__<'py>(
					&self,
					py: Python<'py>,
				) -> Result<Bound<'py, PyBytes>> {
					let inner = &*self.inner.lock();
					let vec = encode_to_vec(
						inner,
						BINCODE_CONFIG,
					)?;

					Ok(PyBytes::new(py, &vec))
				}

				fn __setstate__(
					&self,
					state: Bound<PyBytes>,
				) -> Result<()> {
					let slice = state.as_bytes();
					let (state, _) = decode_from_slice(
						slice,
						BINCODE_CONFIG,
					)?;

					let inner = &mut *self.inner.lock();
					*inner = state;

					Ok(())
				}
			}
		}
	};
}
