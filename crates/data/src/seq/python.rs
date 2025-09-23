use anyhow::Result;
use pyo3::prelude::*;

use std::sync::Arc;

use super::{DnaSeq, FromChars, Seq, parse_str};
use crate::DnaNucleotide;

#[derive(Debug, Clone)]
enum DnaSeqInner {
	Alloc(Box<[DnaNucleotide]>),
	Shared(Arc<[DnaNucleotide]>),
	// TODO: shared slice
}

#[derive(Debug, Clone)]
#[pyclass(name = "DNASeq", module = "aspartik.data", frozen)]
#[repr(transparent)]
pub struct PyDnaSeq(DnaSeqInner);

impl PyDnaSeq {
	pub fn to_shared(&self) -> Self {
		match &self.0 {
			DnaSeqInner::Alloc(boxed) => {
				let arc: Arc<[DnaNucleotide]> =
					boxed.clone().into();
				PyDnaSeq(DnaSeqInner::Shared(arc))
			}

			// already shared
			other => PyDnaSeq(other.clone()),
		}
	}
}

impl PartialEq for PyDnaSeq {
	fn eq(&self, other: &Self) -> bool {
		self.as_slice() == other.as_slice()
	}
}

impl Seq for PyDnaSeq {
	type Character = DnaNucleotide;

	fn as_slice(&self) -> &[DnaNucleotide] {
		match &self.0 {
			DnaSeqInner::Alloc(s) => s.as_slice(),
			DnaSeqInner::Shared(s) => s.as_slice(),
		}
	}
}

impl FromChars for PyDnaSeq {
	fn from_vec(chars: Vec<DnaNucleotide>) -> Self {
		let boxed = Box::<[DnaNucleotide]>::from_vec(chars);
		PyDnaSeq(DnaSeqInner::Alloc(boxed))
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
		Python::attach(|py| Self::new(py, PyDnaSeq::from_vec(chars)))
			.expect("Failed to acquire GIL")
	}
}

#[pymethods]
impl PyDnaSeq {
	#[new]
	fn new(sequence: &str) -> Result<Self> {
		let s: Vec<DnaNucleotide> = parse_str(sequence)?;
		Ok(PyDnaSeq::from_vec(s))
	}

	#[pyo3(name = "to_shared")]
	fn py_to_shared(&self) -> Self {
		self.to_shared()
	}

	fn __str__(&self) -> String {
		self.as_slice().to_string()
	}

	fn __repr__(&self) -> String {
		format!("DNASeq('{}')", self.to_string())
	}

	fn __getitem__(&self, index: usize) -> DnaNucleotide {
		self.as_slice()[index]
	}

	fn __len__(&self) -> usize {
		self.as_slice().len()
	}

	fn complement(&self) -> Self {
		let s = self.as_slice().complement();
		PyDnaSeq::from_vec(s)
	}

	fn reverse_complement(&self) -> Self {
		let s = self.as_slice().complement();
		PyDnaSeq::from_vec(s)
	}

	// TODO: character-generic methods, probably as a macro
}
