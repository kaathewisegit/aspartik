use anyhow::Result;
use pyo3::prelude::*;

use super::{Sequence, parse_str};
use crate::DnaNucleotide;

/// An immutable DNA sequence
///
/// This sequence cannot be mutated.  Instead, its methods can be used for
/// various transformations which will return new sequences.
#[derive(Debug, Clone, PartialEq, Eq)]
#[pyclass(from_py_object, name = "DNASeq", module = "aspartik.data", frozen)]
#[repr(transparent)]
pub struct PyDnaSeq(pub Sequence<DnaNucleotide>);

#[pymethods]
impl PyDnaSeq {
	#[new]
	fn new(sequence: &str) -> Result<Self> {
		let s = parse_str::<DnaNucleotide>(sequence)?;
		Ok(PyDnaSeq(s.into()))
	}

	fn __str__(&self) -> String {
		self.0.to_string()
	}

	fn __repr__(&self) -> String {
		format!("DNASeq('{}')", self.0)
	}

	fn __getitem__(&self, index: usize) -> DnaNucleotide {
		self.0.as_ref()[index]
	}

	fn __len__(&self) -> usize {
		self.0.as_ref().len()
	}

	/// A complementary strand
	///
	/// See `DNANucleotide.complement` for notes on how combined states such
	/// as `DNANucleotide.Weak` and gaps are handled.
	fn complement(&self) -> Self {
		PyDnaSeq(self.0.complement())
	}

	/// Reversed complementary strand
	fn reverse_complement(&self) -> Self {
		PyDnaSeq(self.0.reverse_complement())
	}

	// TODO: character-generic methods, probably as a macro
}
