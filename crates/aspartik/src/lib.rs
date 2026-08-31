use pyo3::prelude::*;

#[pymodule(name = "_aspartik_rust_impl")]
pub mod pymodule {
	#[pymodule_export]
	use {
		b3::pymodule as b3, data::pymodule as data,
		distributions::pymodule as distributions, rng::pymodule as rng,
	};
}
