use anyhow::Result;
use pyo3::prelude::*;

use super::parse_str;
use crate::{
	DnaNucleotide,
	seq::{DisplaySequence, complement, reverse_complement},
};

/// An immutable DNA sequence
///
/// This sequence cannot be mutated.  Instead, its methods can be used for
/// various transformations which will return new sequences.
#[derive(Debug, Clone, PartialEq, Eq)]
#[pyclass(from_py_object, name = "DNASeq", module = "aspartik.data", frozen)]
#[repr(transparent)]
pub struct PyDnaSeq(pub Vec<DnaNucleotide>);

#[pymethods]
impl PyDnaSeq {
	#[new]
	fn new(sequence: &str) -> Result<Self> {
		let s = parse_str::<DnaNucleotide>(sequence)?;
		Ok(PyDnaSeq(s))
	}

	fn __str__(&self) -> String {
		DisplaySequence(&self.0).to_string()
	}

	fn __repr__(&self) -> String {
		format!("DNASeq('{}')", DisplaySequence(&self.0))
	}

	fn __getitem__(&self, index: usize) -> DnaNucleotide {
		self.0[index]
	}

	fn __len__(&self) -> usize {
		self.0.len()
	}

	/// A complementary strand
	///
	/// See `DNANucleotide.complement` for notes on how combined states such
	/// as `DNANucleotide.Weak` and gaps are handled.
	fn complement(&self) -> Self {
		let mut out = self.0.clone();
		complement(&mut out);
		PyDnaSeq(out)
	}

	/// Reversed complementary strand
	fn reverse_complement(&self) -> Self {
		let mut out = self.0.clone();
		reverse_complement(&mut out);
		PyDnaSeq(out)
	}

	// TODO: character-generic methods, probably as a macro
}
