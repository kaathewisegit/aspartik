#[cfg(feature = "python")]
mod fasta;
mod format_readers;
pub mod reader;
pub mod rw;
pub mod sam;
mod str_buf;

pub use format_readers::read_msa_from_fasta;
pub use str_buf::StrBufReader;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "_io_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use fasta::PyFastaDnaReader;

	#[pymodule_export]
	use format_readers::py_read_msa_from_fasta;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
