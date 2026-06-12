use pyo3::prelude::*;

#[pymodule(name = "_aspartik_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use {
		b3::pymodule as b3, data::pymodule as data, io::pymodule as io,
		rng::pymodule as rng, stats::pymodule as stats,
	};

	#[pymodule_init]
	fn init(_: &Bound<'_, PyModule>) -> PyResult<()> {
		Ok(())
	}
}
