pub mod fasta;
pub mod newick;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	let m = PyModule::new(py, "_io_rust_impl")?;

	m.add_class::<newick::python::PyNode>()?;
	m.add_class::<newick::python::PyTree>()?;

	Ok(m)
}
