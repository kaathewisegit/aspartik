use anyhow::Result;
use pyo3::{prelude::*, types::PyType};

use crate::{
	DnaNucleotide, Msa, fasta::python::PyFastaDnaRecord,
	seq::python::PyDnaSeq,
};

#[derive(Debug, Clone)]
#[pyclass(name = "MSA", module = "aspartik.data.msa")]
#[repr(transparent)]
pub struct PyMsa(pub Msa<DnaNucleotide>);

#[pymethods]
impl PyMsa {
	#[classmethod]
	fn from_fasta(
		_cls: Py<PyType>,
		records: Vec<Py<PyFastaDnaRecord>>,
	) -> Result<Self> {
		let msa = Msa::from_fasta(records.into_iter())?;
		Ok(Self(msa))
	}

	#[getter]
	fn num_sites(&self) -> usize {
		self.0.num_sites()
	}

	#[getter]
	fn num_sequences(&self) -> usize {
		self.0.num_sequences()
	}

	fn sequence_name(&self, index: usize) -> String {
		self.0.sequence_name(index).to_owned()
	}

	fn sequence_names(&self) -> Vec<String> {
		self.0.sequence_names().to_vec()
	}

	fn sequence(&self, index: usize) -> PyDnaSeq {
		PyDnaSeq(self.0.sequence(index))
	}

	fn base_frequencies(&self) -> (f64, f64, f64, f64) {
		self.0.base_frequencies().into()
	}
}
