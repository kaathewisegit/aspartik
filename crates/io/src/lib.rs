#[cfg(feature = "python")]
mod fasta;
pub mod rw;
pub mod sam;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "_io_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use fasta::PyFastaDnaReader;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
