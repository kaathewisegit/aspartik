use anyhow::Result;
#[cfg(feature = "python")]
use pyo3::{
	exceptions::{PyBufferError, PyTypeError, PyValueError},
	prelude::*,
	types::{PyBytes, PyString},
};

use std::{
	fs::File,
	io::{Read, Result as IoResult, Write},
	path::Path,
};

use util::{py_bail, py_call_method, py_check_method};

pub enum AnyReader {
	File(File),
	Dynamic(Box<dyn Read + Send>),
	#[cfg(feature = "python")]
	Python(Py<PyAny>),
}

impl AnyReader {
	pub fn from_file<P: AsRef<Path>>(path: P) -> IoResult<Self> {
		File::open(path.as_ref()).map(AnyReader::File)
	}

	pub fn from_dynamic<R>(reader: R) -> Self
	where
		R: Read + Send + 'static,
	{
		AnyReader::Dynamic(Box::new(reader))
	}

	#[cfg(feature = "python")]
	pub fn from_python(obj: Bound<PyAny>) -> Result<Self> {
		py_check_method!(obj, "read");

		Ok(AnyReader::Python(obj.unbind()))
	}
}

impl Read for AnyReader {
	fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
		match self {
			AnyReader::File(f) => f.read(buf),
			AnyReader::Dynamic(reader) => reader.read(buf),
			#[cfg(feature = "python")]
			AnyReader::Python(obj) => py_read(obj, buf),
		}
	}
}

#[cfg(feature = "python")]
fn py_read(obj: &mut Py<PyAny>, buf: &mut [u8]) -> IoResult<usize> {
	let len = buf.len();

	Python::attach(|py| {
		let result = py_call_method!(py, obj, "read", len)?;
		let bytes = if let Ok(bytes) = result.cast_bound::<PyBytes>(py)
		{
			bytes.as_bytes()
		} else if let Ok(s) = result.cast_bound::<PyString>(py) {
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
				"Expected `read` to return at most {} bytes but got {}",
				bytes.len(),
				len,
			);
		}

		buf[..bytes.len()].copy_from_slice(bytes);
		Ok(bytes.len())
	})
}

pub enum AnyWriter {
	File(File),
	Rust(Box<dyn Write + Send>),
	#[cfg(feature = "python")]
	Python(Py<PyAny>),
}

impl AnyWriter {
	pub fn from_dynamic<W>(writer: W) -> Self
	where
		W: Write + Send + 'static,
	{
		AnyWriter::Rust(Box::new(writer))
	}

	#[cfg(feature = "python")]
	pub fn from_python(obj: Bound<PyAny>) -> Result<Self> {
		py_check_method!(obj, "write");

		Ok(AnyWriter::Python(obj.unbind()))
	}
}

impl Write for AnyWriter {
	fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
		match self {
			AnyWriter::File(f) => f.write(buf),
			AnyWriter::Rust(writer) => writer.write(buf),
			#[cfg(feature = "python")]
			AnyWriter::Python(obj) => py_write(obj, buf),
		}
	}

	fn flush(&mut self) -> IoResult<()> {
		match self {
			AnyWriter::File(f) => f.flush(),
			AnyWriter::Rust(writer) => writer.flush(),
			#[cfg(feature = "python")]
			AnyWriter::Python(obj) => py_flush(obj),
		}
	}
}

#[cfg(feature = "python")]
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
				"Expected `write` to return a positive number, got `{}`",
				num,
			);
		};

		Ok(num)
	})
}

#[cfg(feature = "python")]
fn py_flush(obj: &mut Py<PyAny>) -> IoResult<()> {
	Python::attach(|py| {
		py_call_method!(py, obj, "flush")?;
		Ok(())
	})
}
