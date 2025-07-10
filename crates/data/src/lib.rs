mod nucleotides;
mod phred;
pub mod seq;

pub use nucleotides::{DnaNucleotide, DnaNucleotideError};
pub use phred::Phred;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	use util::py_make_submodule;
	let m = py_make_submodule!(py, "_data_rust_impl");

	m.add_class::<DnaNucleotide>()?;
	m.add_class::<DnaNucleotideError>()?;

	m.add_class::<Phred>()?;

	use seq::python::PyDnaSeq;
	m.add_class::<PyDnaSeq>()?;

	Ok(m)
}
