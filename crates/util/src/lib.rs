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
