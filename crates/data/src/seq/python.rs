use anyhow::Result;
use pyo3::prelude::*;

use super::{parse_str, DnaSeq, Seq};
use crate::DnaNucleotide;

#[derive(Debug, Clone)]
#[pyclass(name = "DNASeq", module = "aspartik.data", frozen)]
pub struct PyDnaSeq(pub DnaSeq);

impl From<DnaSeq> for PyDnaSeq {
	fn from(value: DnaSeq) -> Self {
		PyDnaSeq(value)
	}
}

impl From<PyDnaSeq> for DnaSeq {
	fn from(value: PyDnaSeq) -> Self {
		value.0
	}
}

#[pymethods]
impl PyDnaSeq {
	#[new]
	fn new(sequence: &str) -> Result<Self> {
		Ok(PyDnaSeq(parse_str(sequence)?))
	}

	fn __str__(&self) -> String {
		self.0.to_string()
	}

	fn __repr__(&self) -> String {
		format!("DNASeq('{}')", self.0)
	}

	fn __getitem__(&self, index: usize) -> DnaNucleotide {
		self.0[index]
	}

	fn __len__(&self) -> usize {
		self.0.len()
	}

	fn complement(&self) -> Self {
		PyDnaSeq(self.0.complement())
	}

	fn reverse_complement(&self) -> Self {
		PyDnaSeq(self.0.reverse_complement())
	}

	// TODO: character-generic methods, probably as a macro
}
