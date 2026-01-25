use anyhow::Result;
use pyo3::{prelude::*, types::PyType};

use crate::{
	DnaNucleotide, Msa, fasta::python::PyFastaDnaRecord,
	seq::python::PyDnaSeq,
};

/// DNA multiple sequence alignment
///
/// A set of sequences of the same length along with their names.
#[derive(Debug, Clone)]
#[pyclass(name = "MSA", module = "aspartik.data.msa", frozen)]
#[repr(transparent)]
pub struct PyMsa(pub Msa<DnaNucleotide>);

#[pymethods]
impl PyMsa {
	/// Constructs an MSA from a list of FASTA records
	#[classmethod]
	fn from_fasta(
		_cls: Py<PyType>,
		records: Vec<Py<PyFastaDnaRecord>>,
	) -> Result<Self> {
		let msa = Msa::from_fasta(records.into_iter())?;
		Ok(Self(msa))
	}

	/// Total number of sites, including gaps
	#[getter]
	fn num_sites(&self) -> usize {
		self.0.num_sites()
	}

	/// Number of sequences
	#[getter]
	fn num_sequences(&self) -> usize {
		self.0.num_sequences()
	}

	/// The name of the `index`'th sequence
	fn sequence_name(&self, index: usize) -> String {
		self.0.sequence_name(index).to_owned()
	}

	/// A list with all of the sequence names
	fn sequence_names(&self) -> Vec<String> {
		self.0.sequence_names().to_vec()
	}

	/// `index`'th sequence
	fn sequence(&self, index: usize) -> PyDnaSeq {
		PyDnaSeq(self.0.sequence(index))
	}

	/// The shares each DNA base takes up in the total alignment
	///
	/// Compound bases such as `NotGuanine` are split equiproportionally
	/// between their components.  Gaps are counted the same way as `Any`.
	/// The components of the resulting tuple should always add up to almost
	/// 1, taking floating point precision limitations into account.
	fn base_frequencies(&self) -> (f64, f64, f64, f64) {
		self.0.base_frequencies().into()
	}
}
