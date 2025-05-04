use pyo3::prelude::*;

#[pymodule(name = "_aspartik_rust_impl")]
fn aspartik(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
	let b3 = PyModule::new(py, "_b3_rust_impl")?;
	b3::pymodule(py, &b3)?;
	m.add_submodule(&b3)?;

	let rng = PyModule::new(py, "_rng_rust_impl")?;
	rng::pymodule(py, &rng)?;
	m.add_submodule(&rng)?;

	let stats = PyModule::new(py, "_stats_rust_impl")?;
	stats::pymodule(py, &stats)?;
	m.add_submodule(&stats)?;

	m.add_submodule(&data::pymodule(py)?)?;

	Ok(())
}
