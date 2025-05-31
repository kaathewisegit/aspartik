use anyhow::Result;
use pyo3::prelude::*;

use super::{parse_str, DnaSeq, FromChars, Seq};
use crate::DnaNucleotide;

#[derive(Debug, Clone)]
#[pyclass(name = "DNASeq", module = "aspartik.data", frozen)]
pub struct PyDnaSeq(pub Box<[DnaNucleotide]>);

impl Seq for PyDnaSeq {
	type Character = DnaNucleotide;

	fn as_slice(&self) -> &[DnaNucleotide] {
		self.0.as_slice()
	}
}

impl FromChars for PyDnaSeq {
	fn from_vec(chars: Vec<DnaNucleotide>) -> Self {
		PyDnaSeq(Box::<[DnaNucleotide]>::from_vec(chars))
	}
}

impl Seq for Py<PyDnaSeq> {
	type Character = DnaNucleotide;

	fn as_slice(&self) -> &[DnaNucleotide] {
		self.get().as_slice()
	}
}

impl FromChars for Py<PyDnaSeq> {
	fn from_vec(chars: Vec<DnaNucleotide>) -> Self {
		Python::with_gil(|py| Self::new(py, PyDnaSeq::from_vec(chars)))
			.expect("Failed to acquire GIL")
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
		format!("DNASeq('{}')", self.to_string())
	}

	fn __getitem__(&self, index: usize) -> DnaNucleotide {
		self.0[index]
	}

	fn __len__(&self) -> usize {
		self.0.len()
	}

	fn complement(&self) -> Self {
		PyDnaSeq(self.0.complement().into())
	}

	fn reverse_complement(&self) -> Self {
		PyDnaSeq(self.0.reverse_complement().into())
	}

	// TODO: character-generic methods, probably as a macro
}
