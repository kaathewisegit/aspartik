use pyo3::prelude::*;

use super::*;
use data::{seq::PyDnaSeq, DnaNucleotide};

#[pyclass(name = "FASTADNARecord", module = "aspartik.io.fasta", frozen)]
pub struct PyFastaDnaRecord(Record<DnaNucleotide>);

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
