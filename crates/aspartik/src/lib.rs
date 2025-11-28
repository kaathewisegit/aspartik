use pyo3::prelude::*;

#[pymodule(name = "_aspartik_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use b3::pymodule as b3;
	#[pymodule_export]
	use data::pymodule as data;
	#[pymodule_export]
	use io::pymodule as io;
	#[pymodule_export]
	use logger::pymodule as logger;
	#[pymodule_export]
	use math::pymodule as math;
	#[pymodule_export]
	use rng::pymodule as rng;
	#[pymodule_export]
	use stats::pymodule as stats;

	#[pymodule_init]
	fn init(_: &Bound<'_, PyModule>) -> PyResult<()> {
		Ok(())
	}
}
