use anyhow::Result;
use pyo3::prelude::*;

use super::DnaSeq;
use crate::DnaNucleotide;

#[pyclass(name = "DNASeq", module = "aspartik.data", frozen)]
pub struct PyDnaSeq(pub DnaSeq);

impl From<DnaSeq> for PyDnaSeq {
	fn from(value: DnaSeq) -> Self {
		PyDnaSeq(value)
	}
}

#[pymethods]
impl PyDnaSeq {
	#[new]
	#[pyo3(signature = (sequence = None))]
	fn new(sequence: Option<&str>) -> Result<Self> {
		let seq = sequence
			.map(DnaSeq::try_from)
			.unwrap_or_else(|| Ok(DnaSeq::new()))?;

		Ok(PyDnaSeq(seq))
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

	fn complement(&self) -> Self {
		PyDnaSeq(self.0.complement())
	}

	fn reverse_complement(&self) -> Self {
		PyDnaSeq(self.0.reverse_complement())
	}

	// TODO: character-generic methods, probably as a macro
}
