#![forbid(unsafe_code)]

pub mod consts;
pub mod float;
pub mod function;
mod ranged;
pub mod tolerance;

pub use ranged::{Positive, Probability};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "_math_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use function::erf::erf;
	#[pymodule_export]
	use function::erf::erf_inv;
	#[pymodule_export]
	use function::erf::erfc;
	#[pymodule_export]
	use function::erf::erfc_inv;

	#[pymodule_export]
	use function::exponential::ei;

	#[pymodule_export]
	use function::factorial::binomial;
	#[pymodule_export]
	use function::factorial::factorial;
	#[pymodule_export]
	use function::factorial::ln_binomial;
	#[pymodule_export]
	use function::factorial::ln_factorial;

	#[pymodule_export]
	use function::gamma::digamma;
	#[pymodule_export]
	use function::gamma::digamma_inv;
	#[pymodule_export]
	use function::gamma::gamma;
	#[pymodule_export]
	use function::gamma::gamma_li;
	#[pymodule_export]
	use function::gamma::gamma_lr;
	#[pymodule_export]
	use function::gamma::gamma_ui;
	#[pymodule_export]
	use function::gamma::gamma_ur;
	#[pymodule_export]
	use function::gamma::ln_gamma;

	#[pymodule_export]
	use function::harmonic::generalized_harmonic;
	#[pymodule_export]
	use function::harmonic::harmonic;

	#[pymodule_export]
	use function::logistic::logistic;
	#[pymodule_export]
	use function::logistic::logit;

	#[pymodule_export]
	use tolerance::is_close;

	#[pymodule_export]
	use float::exponent;
	#[pymodule_export]
	use float::exponent_bits;
	#[pymodule_export]
	use float::mantissa;
	#[pymodule_export]
	use float::mantissa_bits;
	#[pymodule_export]
	use float::sign;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
