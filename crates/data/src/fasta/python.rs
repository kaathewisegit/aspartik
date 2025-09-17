use pyo3::prelude::*;

use super::Record;
use crate::seq::python::PyDnaSeq;

#[pyclass(name = "DNARecord", module = "aspartik.data.fasta", frozen)]
pub struct PyFastaDnaRecord(pub Record<Py<PyDnaSeq>>);

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
