pub mod fasta;
mod msa;
pub mod newick;
mod nucleotides;
mod phred;
pub mod seq;

pub use msa::Msa;
#[cfg(feature = "python")]
pub use msa::python::PyMsa;
pub use nucleotides::DnaNucleotide;
pub use phred::Phred;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "_data_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use fasta::python::PyFastaDnaRecord;

	#[pymodule_export]
	use msa::python::PyMsa;

	#[pymodule_export]
	use newick::python::PyTree;

	#[pymodule_export]
	use DnaNucleotide;

	#[pymodule_export]
	use Phred;

	#[pymodule_export]
	use seq::python::PyDnaSeq;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
