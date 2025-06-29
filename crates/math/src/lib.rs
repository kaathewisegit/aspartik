#![forbid(unsafe_code)]

pub mod consts;
pub mod function;
pub mod tolerance;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	let m = PyModule::new(py, "_math_rust_impl")?;
	m.add_function(wrap_pyfunction!(function::erf::erf, &m)?)?;
	m.add_function(wrap_pyfunction!(function::erf::erf_inv, &m)?)?;
	m.add_function(wrap_pyfunction!(function::erf::erfc, &m)?)?;
	m.add_function(wrap_pyfunction!(function::erf::erfc_inv, &m)?)?;

	m.add_function(wrap_pyfunction!(tolerance::is_close, &m)?)?;

	Ok(m)
}
