use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::fs::File;

use super::{FastaReader, Record};
use data::{seq::python::PyDnaSeq, DnaNucleotide};

#[pyclass(name = "FASTADNARecord", module = "aspartik.io.fasta", frozen)]
pub struct PyFastaDnaRecord(Record<DnaNucleotide>);

impl From<Record<DnaNucleotide>> for PyFastaDnaRecord {
	fn from(value: Record<DnaNucleotide>) -> Self {
		Self(value)
	}
}

#[pymethods]
impl PyFastaDnaRecord {
	#[getter]
	fn sequence(&self) -> PyDnaSeq {
		// TODO: perhaps there's a way to avoid cloning.  Probably by
		// reimplementing `Seq`'s methods.
		self.0.sequence().to_owned().into()
	}

	#[getter]
	fn raw_description(&self) -> String {
		self.0.raw_description().to_owned()
	}

	#[getter]
	fn description(&self) -> String {
		self.0.description().to_owned()
	}
}

#[pyclass(name = "FASTADNAReader", module = "aspartik.io.fasta", frozen)]
pub struct PyFastaDnaReader {
	inner: Mutex<FastaReader<DnaNucleotide, File>>,
}

#[pymethods]
impl PyFastaDnaReader {
	#[new]
	fn new(path: &str) -> Result<Self> {
		let file = File::open(path)?;
		let reader = FastaReader::new(file);
		Ok(Self {
			inner: Mutex::new(reader),
		})
	}

	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__(&self) -> Option<Result<PyFastaDnaRecord>> {
		self.inner.lock().next().map(|r| r.map(|r| r.into()))
	}
}
