use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyType};

use std::{
	fs::File,
	io::{BufRead, BufReader},
};

use crate::rw::AnyReader;
use data::{
	fasta::{FastaParser, python::PyFastaDnaRecord},
	seq::python::PyDnaSeq,
};

#[pyclass(name = "FastaReader", module = "aspartik.io", frozen)]
pub struct PyFastaDnaReader {
	parser: Mutex<FastaParser<Py<PyDnaSeq>>>,
	reader: Mutex<BufReader<AnyReader>>,
}

macro_rules! bubble {
	($e: expr) => {
		match $e {
			Ok(out) => out,
			Err(e) => return Some(Err(e.into())),
		}
	};
}

#[pymethods]
impl PyFastaDnaReader {
	#[new]
	fn new(obj: Bound<PyAny>) -> Result<Self> {
		let reader = AnyReader::from_python(obj)?;
		let buf_reader = BufReader::new(reader);
		Ok(Self {
			parser: Mutex::new(FastaParser::new()),
			reader: Mutex::new(buf_reader),
		})
	}

	#[classmethod]
	fn from_file(_cls: Py<PyType>, path: &str) -> Result<Self> {
		let file = File::open(path)?;
		let reader = AnyReader::from_dynamic(file);
		let buf_reader = BufReader::new(reader);
		Ok(Self {
			parser: Mutex::new(FastaParser::new()),
			reader: Mutex::new(buf_reader),
		})
	}

	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__(&self) -> Option<Result<PyFastaDnaRecord>> {
		let parser = &mut *self.parser.lock();
		let reader = &mut *self.reader.lock();

		let mut buf = String::new();

		loop {
			buf.clear();
			if bubble!(reader.read_line(&mut buf)) == 0 {
				break;
			}
			trim_line_end(&mut buf);

			if let Some(record) =
				bubble!(parser.read_line(Some(&buf)))
			{
				return Some(Ok(PyFastaDnaRecord(record)));
			}
		}

		let record = bubble!(parser.read_line(None));

		record.map(PyFastaDnaRecord).map(Ok)
	}
}

fn trim_line_end(line: &mut String) {
	if line.ends_with('\n') {
		line.pop();
		if line.ends_with('\r') {
			line.pop();
		}
	}
}
