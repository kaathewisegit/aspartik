#![forbid(unsafe_code)]

pub mod consts;
pub mod function;
pub mod tolerance;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	let m = PyModule::new(py, "_math_rust_impl")?;

	use function::erf::*;
	m.add_function(wrap_pyfunction!(erf, &m)?)?;
	m.add_function(wrap_pyfunction!(erf_inv, &m)?)?;
	m.add_function(wrap_pyfunction!(erfc, &m)?)?;
	m.add_function(wrap_pyfunction!(erfc_inv, &m)?)?;

	use function::exponential::*;
	m.add_function(wrap_pyfunction!(ei, &m)?)?;

	use function::factorial::*;
	m.add_function(wrap_pyfunction!(factorial, &m)?)?;
	m.add_function(wrap_pyfunction!(ln_factorial, &m)?)?;
	m.add_function(wrap_pyfunction!(binomial, &m)?)?;
	m.add_function(wrap_pyfunction!(ln_binomial, &m)?)?;

	use function::gamma::*;
	m.add_function(wrap_pyfunction!(gamma, &m)?)?;
	m.add_function(wrap_pyfunction!(ln_gamma, &m)?)?;
	m.add_function(wrap_pyfunction!(gamma_ui, &m)?)?;
	m.add_function(wrap_pyfunction!(gamma_li, &m)?)?;
	m.add_function(wrap_pyfunction!(gamma_ur, &m)?)?;
	m.add_function(wrap_pyfunction!(gamma_lr, &m)?)?;
	m.add_function(wrap_pyfunction!(digamma, &m)?)?;
	m.add_function(wrap_pyfunction!(digamma_inv, &m)?)?;

	m.add_function(wrap_pyfunction!(tolerance::is_close, &m)?)?;

	Ok(m)
}
