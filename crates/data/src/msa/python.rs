use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;

use crate::{DnaNucleotide, Msa};

#[derive(Debug)]
#[pyclass]
#[repr(transparent)]
pub struct PyMsa {
	inner: Mutex<Msa<DnaNucleotide>>,
}

impl PyMsa {
	fn inner(&self) -> MutexGuard<'_, Msa<DnaNucleotide>> {
		self.inner.lock()
	}
}

impl Clone for PyMsa {
	fn clone(&self) -> Self {
		let msa = self.inner().clone();

		PyMsa {
			inner: Mutex::new(msa),
		}
	}
}

#[pymethods]
impl PyMsa {
	fn deduplicate(&self) {
		self.inner().deduplicate()
	}
}
