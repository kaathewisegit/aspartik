use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::{
	fs::File,
	io::{BufRead, BufReader},
};

use super::{FastaParser, Record};
use data::seq::python::PyDnaSeq;

#[pyclass(name = "DNARecord", module = "aspartik.io.fasta", frozen)]
pub struct PyFastaDnaRecord(Record<Py<PyDnaSeq>>);

#[pymethods]
impl PyFastaDnaRecord {
	#[new]
	fn new(description: String, sequence: Py<PyDnaSeq>) -> Self {
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

		self.0.description == other.0.description
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
	reader: Mutex<BufReader<File>>,
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
	fn new(path: &str) -> Result<Self> {
		let file = File::open(path)?;
		let reader = BufReader::new(file);
		Ok(Self {
			parser: Mutex::new(FastaParser::new()),
			reader: Mutex::new(reader),
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
