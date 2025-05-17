use pyo3::prelude::*;

use std::sync::{Arc, Mutex, MutexGuard};

use super::*;
use data::DnaNucleotide;

#[cfg_attr(
	feature = "python",
	pyclass(name = "FASTADNARecord", module = "aspartik.io.fasta", frozen)
)]
pub struct PyFastaDnaRecord {
	inner: Arc<Mutex<Record<DnaNucleotide>>>,
}

impl PyFastaDnaRecord {
	pub fn inner(&self) -> MutexGuard<Record<DnaNucleotide>> {
		self.inner.lock().unwrap()
	}
}

#[pymethods]
impl PyFastaDnaRecord {
	#[getter]
	fn raw_description(&self) -> String {
		self.inner().raw_description().to_owned()
	}

	#[getter]
	fn description(&self) -> String {
		self.inner().description().to_owned()
	}
}
