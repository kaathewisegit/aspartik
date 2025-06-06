use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::{fs::File, io::BufReader};

use super::{FastaReader, Record};
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
	inner: Mutex<FastaReader<Py<PyDnaSeq>, BufReader<File>>>,
}

#[pymethods]
impl PyFastaDnaReader {
	#[new]
	fn new(path: &str) -> Result<Self> {
		let file = File::open(path)?;
		let reader = FastaReader::from_file(file);
		Ok(Self {
			inner: Mutex::new(reader),
		})
	}

	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__(&self) -> Option<Result<PyFastaDnaRecord>> {
		let record = self.inner.lock().next()?;
		Some(record.map(PyFastaDnaRecord))
	}
}
