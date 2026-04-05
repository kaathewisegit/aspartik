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
	use function::{
		erf::{erf, erf_inv, erfc, erfc_inv},
		exponential::ei,
		factorial::{binomial, factorial, ln_binomial, ln_factorial},
		gamma::{
			digamma, digamma_inv, gamma, gamma_li, gamma_lr,
			gamma_ui, gamma_ur, ln_gamma,
		},
		harmonic::{generalized_harmonic, harmonic},
		logistic::{logistic, logit},
	};

	#[pymodule_export]
	use tolerance::is_close;

	#[pymodule_export]
	use float::{exponent_bits, mantissa_bits, sign};

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
