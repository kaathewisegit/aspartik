pub mod fasta;
mod msa;
pub mod newick;
mod nucleotides;
mod phred;
pub mod seq;

pub use msa::Msa;
pub use nucleotides::DnaNucleotide;
pub use phred::Phred;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	use util::py_make_submodule;
	let m = py_make_submodule!(py, "_data_rust_impl");

	// fasta
	m.add_class::<fasta::python::PyFastaDnaRecord>()?;

	// msa
	m.add_class::<msa::python::PyMsa>()?;

	// newick
	m.add_class::<newick::python::PyTree>()?;

	// nucleotides
	m.add_class::<DnaNucleotide>()?;

	// phred
	m.add_class::<Phred>()?;

	// seq
	m.add_class::<seq::python::PyDnaSeq>()?;

	Ok(m)
}
