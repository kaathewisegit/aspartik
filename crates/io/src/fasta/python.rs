use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyType};

use std::{
	fs::File,
	io::{BufRead, BufReader},
};

use super::{FastaParser, Record};
use crate::rw::AnyReader;
use data::seq::python::PyDnaSeq;

#[pyclass(name = "DNARecord", module = "aspartik.io.fasta", frozen)]
pub struct PyFastaDnaRecord(Record<Py<PyDnaSeq>>);

#[pymethods]
impl PyFastaDnaRecord {
	#[new]
	fn new(mut description: String, sequence: Py<PyDnaSeq>) -> Self {
		if !description.starts_with('>') {
			description.insert(0, '>');
		}
		let record = Record::new(description, sequence);
		Self(record)
	}

	#[getter]
	fn sequence(&self, py: Python) -> Py<PyDnaSeq> {
		// TODO: perhaps there's a way to avoid cloning.  Probably by
		// reimplementing `Seq`'s methods.
		self.0.sequence().clone_ref(py)
	}

	#[getter]
	fn raw_description(&self) -> String {
		self.0.raw_description().to_owned()
	}

	#[getter]
	fn description(&self) -> String {
		self.0.description().to_owned()
	}

	#[getter]
	fn id(&self) -> String {
		self.0.id().to_string()
	}

	fn __eq__(&self, other: &Self) -> bool {
		let self_seq = self.0.seq.get();
		let other_seq = other.0.seq.get();

		self.0.raw_description == other.0.raw_description
			&& self_seq == other_seq
	}

	fn __str__(&self) -> String {
		self.0.to_string()
	}

	fn __repr__(&self) -> String {
		format!(
			r#"DNARecord({:?}, DNASeq("{}"))"#,
			self.0.raw_description(),
			self.0.sequence(),
		)
	}
}

#[pyclass(name = "DNAReader", module = "aspartik.io.fasta", frozen)]
pub struct PyFastaDnaReader {
	parser: Mutex<FastaParser<Py<PyDnaSeq>>>,
	// TODO: universal reader struct for IO
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
	fn new(obj: PyObject) -> Self {
		let reader = AnyReader::from_python(obj);
		let buf_reader = BufReader::new(reader);
		Self {
			parser: Mutex::new(FastaParser::new()),
			reader: Mutex::new(buf_reader),
		}
	}

	#[classmethod]
	fn from_file(_cls: Py<PyType>, path: &str) -> Result<Self> {
		let file = File::open(path)?;
		let reader = AnyReader::from_rust(file);
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
