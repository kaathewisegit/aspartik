use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::{
	DnaNucleotide, Msa, fasta::python::PyFastaDnaRecord,
	seq::python::PyDnaSeq,
};

#[derive(Debug)]
#[pyclass(name = "MSA", module = "aspartik.data.msa")]
#[repr(transparent)]
pub struct PyMsa {
	inner: Mutex<Msa<DnaNucleotide>>,
}

impl PyMsa {
	fn new(msa: Msa<DnaNucleotide>) -> Self {
		PyMsa {
			inner: Mutex::new(msa),
		}
	}

	fn inner(&self) -> MutexGuard<'_, Msa<DnaNucleotide>> {
		self.inner.lock()
	}
}

impl Clone for PyMsa {
	fn clone(&self) -> Self {
		let msa = self.inner().clone();
		Self::new(msa)
	}
}

#[pymethods]
impl PyMsa {
	#[classmethod]
	fn from_fasta(
		_cls: Py<PyType>,
		records: Vec<Py<PyFastaDnaRecord>>,
	) -> Result<Self> {
		let msa = Msa::from_fasta(records.into_iter())?;
		Ok(Self::new(msa))
	}

	#[getter]
	fn num_sites(&self) -> usize {
		self.inner().num_sites()
	}

	#[getter]
	fn num_sequences(&self) -> usize {
		self.inner().num_sequences()
	}

	pub fn sequence_name(&self, index: usize) -> String {
		self.inner().sequence_name(index).to_owned()
	}

	pub fn sequence(&self, index: usize) -> PyDnaSeq {
		let seq = self.inner().sequence(index);
		PyDnaSeq(seq)
	}

	fn deduplicate(&self) {
		self.inner().deduplicate()
	}
}
