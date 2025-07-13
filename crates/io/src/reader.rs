use pyo3::{
	exceptions::{PyBufferError, PyTypeError},
	prelude::*,
	types::{PyBytes, PyString},
};

use std::io::{Read, Result as IoResult};

use util::{py_bail, py_call_method};

pub enum AnyReader {
	Rust(Box<dyn Read + Send>),
	Python(PyObject),
}

impl AnyReader {
	pub fn from_rust<R>(reader: R) -> Self
	where
		R: Read + Send + 'static,
	{
		AnyReader::Rust(Box::new(reader))
	}

	pub fn from_python(obj: PyObject) -> Self {
		AnyReader::Python(obj)
	}
}

impl Read for AnyReader {
	fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
		match self {
			AnyReader::Rust(reader) => reader.read(buf),
			AnyReader::Python(obj) => py_read(obj, buf),
		}
	}
}

fn py_read(obj: &mut PyObject, buf: &mut [u8]) -> IoResult<usize> {
	let len = buf.len();

	Python::with_gil(|py| {
		let result = py_call_method!(py, obj, "read", len)?;
		let bytes = if let Ok(bytes) =
			result.downcast_bound::<PyBytes>(py)
		{
			bytes.as_bytes()
		} else if let Ok(s) = result.downcast_bound::<PyString>(py) {
			s.to_str()?.as_bytes()
		} else {
			py_bail!(
				PyTypeError,
				"Expected the `read` method on `{}` to return `str` or `bytes`, got `{}`",
				obj.bind(py).repr()?,
				result.bind(py).get_type().name()?,
			);
		};

		if bytes.len() > len {
			py_bail!(
				PyBufferError,
				"`read` was required to returned at most {} bytes but returned {}",
				bytes.len(),
				len,
			);
		}

		buf[..bytes.len()].copy_from_slice(bytes);
		Ok(bytes.len())
	})
}
