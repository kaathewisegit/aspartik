mod nucleotides;
pub mod seq;

pub use nucleotides::{DnaNucleotide, DnaNucleotideError};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	let m = PyModule::new(py, "_data_rust_impl")?;

	m.add_class::<DnaNucleotide>()?;
	m.add_class::<DnaNucleotideError>()?;

	use seq::python::PyDnaSeq;
	m.add_class::<PyDnaSeq>()?;

	Ok(m)
}
