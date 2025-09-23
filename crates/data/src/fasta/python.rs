use pyo3::prelude::*;

use super::Record;
use crate::{DnaNucleotide, seq::python::PyDnaSeq};

#[derive(Debug, PartialEq, Eq)]
#[pyclass(name = "DNARecord", module = "aspartik.data.fasta", frozen, eq)]
pub struct PyFastaDnaRecord(pub Record<DnaNucleotide>);

#[pymethods]
impl PyFastaDnaRecord {
	#[new]
	fn new(mut description: String, sequence: Py<PyDnaSeq>) -> Self {
		if !description.starts_with('>') {
			description.insert(0, '>');
		}
		let record = Record::new(description, sequence.get().0.clone());
		Self(record)
	}

	#[getter]
	fn sequence(&self) -> PyDnaSeq {
		PyDnaSeq(self.0.seq.clone())
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
