#[cfg(feature = "python")]
mod fasta;
pub mod newick;
#[cfg(feature = "python")]
pub mod rw;
pub mod sam;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	use util::py_make_submodule;
	let m = py_make_submodule!(py, "_io_rust_impl");

	m.add_class::<fasta::PyFastaDnaReader>()?;

	m.add_class::<newick::python::PyNode>()?;
	m.add_class::<newick::python::PyTree>()?;

	Ok(m)
}
