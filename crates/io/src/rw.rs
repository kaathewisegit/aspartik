//! Read/write types which work for both Python and Rust
//!
//! These allow creating readers/writers which abstract the underlying
//! implementation, be it from Python or Rust.
//!
//! Rust types are boxed.  They must be [`Send`], because they are typically
//! embedded in Python classes, which can be sent accross threads.

// XXX: validate Python types

use pyo3::{
	exceptions::{PyBufferError, PyTypeError, PyValueError},
	prelude::*,
	types::{PyBytes, PyString},
};

use std::io::{Read, Result as IoResult, Write};

use util::{py_bail, py_call_method};

pub enum AnyReader {
	Rust(Box<dyn Read + Send>),
	Python(Py<PyAny>),
}

impl AnyReader {
	pub fn from_rust<R>(reader: R) -> Self
	where
		R: Read + Send + 'static,
	{
		AnyReader::Rust(Box::new(reader))
	}

	pub fn from_python(obj: Py<PyAny>) -> Self {
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

fn py_read(obj: &mut Py<PyAny>, buf: &mut [u8]) -> IoResult<usize> {
	let len = buf.len();

	Python::attach(|py| {
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

pub enum AnyWriter {
	Rust(Box<dyn Write + Send>),
	Python(Py<PyAny>),
}

impl AnyWriter {
	pub fn from_rust<W>(writer: W) -> Self
	where
		W: Write + Send + 'static,
	{
		AnyWriter::Rust(Box::new(writer))
	}

	pub fn from_python(obj: Py<PyAny>) -> Self {
		AnyWriter::Python(obj)
	}
}

impl Write for AnyWriter {
	fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
		match self {
			AnyWriter::Rust(writer) => writer.write(buf),
			AnyWriter::Python(obj) => py_write(obj, buf),
		}
	}

	fn flush(&mut self) -> IoResult<()> {
		match self {
			AnyWriter::Rust(writer) => writer.flush(),
			AnyWriter::Python(obj) => py_flush(obj),
		}
	}
}

fn py_write(obj: &mut Py<PyAny>, buf: &[u8]) -> IoResult<usize> {
	Python::attach(|py| {
		let result = py_call_method!(py, obj, "write", buf)?;
		let Ok(num) = result.extract::<isize>(py) else {
			py_bail!(
				PyTypeError,
				"Expected `write` to return `int`, got `{}`",
				result.bind(py).get_type().name()?,
			);
		};

		let Ok(num): Result<usize, _> = num.try_into() else {
			py_bail!(
				PyValueError,
				"Expected `write` to a positive number, got `{}`",
				num,
			);
		};

		Ok(num)
	})
}

fn py_flush(obj: &mut Py<PyAny>) -> IoResult<()> {
	Python::attach(|py| {
		py_call_method!(py, obj, "flush")?;
		Ok(())
	})
}
