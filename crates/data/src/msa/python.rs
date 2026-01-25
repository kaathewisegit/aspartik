use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::{
	DnaNucleotide, Msa, fasta::python::PyFastaDnaRecord,
	seq::python::PyDnaSeq,
};
use util::py_pickle_state_impl;

/// DNA multiple sequence alignment
///
/// A set of sequences of the same length along with their names.
#[derive(Debug)]
#[pyclass(name = "MSA", module = "aspartik.data.msa", frozen, eq)]
#[repr(transparent)]
pub struct PyMsa {
	inner: Mutex<Msa<DnaNucleotide>>,
}

impl PyMsa {
	pub fn inner(&self) -> MutexGuard<'_, Msa<DnaNucleotide>> {
		self.inner.lock()
	}
}

impl PartialEq for PyMsa {
	fn eq(&self, other: &Self) -> bool {
		let a = &*self.inner();
		let b = &*other.inner();
		a == b
	}
}

#[pymethods]
impl PyMsa {
	/// Constructs an MSA from a list of FASTA records
	#[classmethod]
	fn from_fasta(
		_cls: Py<PyType>,
		records: Vec<Py<PyFastaDnaRecord>>,
	) -> Result<Self> {
		let msa = Msa::from_fasta(records.into_iter())?;
		Ok(Self {
			inner: Mutex::new(msa),
		})
	}

	/// Total number of sites, including gaps
	#[getter]
	fn num_sites(&self) -> usize {
		self.inner().num_sites()
	}

	/// Number of sequences
	#[getter]
	fn num_sequences(&self) -> usize {
		self.inner().num_sequences()
	}

	/// The name of the `index`'th sequence
	fn sequence_name(&self, index: usize) -> String {
		self.inner().sequence_name(index).to_owned()
	}

	/// A list with all of the sequence names
	fn sequence_names(&self) -> Vec<String> {
		self.inner().sequence_names().to_vec()
	}

	/// `index`'th sequence
	fn sequence(&self, index: usize) -> PyDnaSeq {
		PyDnaSeq(self.inner().sequence(index))
	}

	/// The shares each DNA base takes up in the total alignment
	///
	/// Compound bases such as `NotGuanine` are split equiproportionally
	/// between their components.  Gaps are counted the same way as `Any`.
	/// The components of the resulting tuple should always add up to almost
	/// 1, taking floating point precision limitations into account.
	fn base_frequencies(&self) -> (f64, f64, f64, f64) {
		self.inner().base_frequencies().into()
	}
}

py_pickle_state_impl!(PyMsa, _msa_pickle_impl);
