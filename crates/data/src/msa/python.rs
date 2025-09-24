use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::{DnaNucleotide, Msa, fasta::python::PyFastaDnaRecord};

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

	fn deduplicate(&self) {
		self.inner().deduplicate()
	}
}
