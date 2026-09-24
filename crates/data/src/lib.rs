mod aa;
pub mod fasta;
mod msa;
pub mod nexus;
mod nucleotides;
mod parser;
mod phred;
pub mod seq;
pub mod tree;

pub use aa::AminoAcid;
pub use msa::Msa;
#[cfg(feature = "python")]
pub use msa::python::PyMsa;
pub use nucleotides::DnaNucleotide;
pub use parser::Parser;
pub use phred::Phred;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "_data_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use crate::{
		AminoAcid, DnaNucleotide, Phred,
		fasta::python::PyFastaDnaRecord,
		msa::python::PyMsa,
		seq::python::PyDnaSeq,
		tree::python::{PyBinaryTree, PySvgOptions, PyTreeBuilder},
		tree::{Internal, Leaf, Node},
	};

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);
		m.add_function(wrap_pyfunction!(
			crate::tree::python::py_robinson_foulds_matrix,
			m
		)?)?;
		m.add_function(wrap_pyfunction!(
			crate::tree::python::py_triplet_distance_matrix,
			m
		)?)?;

		Ok(())
	}
}
