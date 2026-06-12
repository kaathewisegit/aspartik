//! This crate aims is a fork of the [`statrs`][s] statistical utilities crate.
//! Currently it only provides distributions with support for sampling, a number
//! of statistics, and using all of those in Python.
//!
//! [s]: https://lib.rs/crates/statrs

#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unused)]

#[macro_use]
pub mod distribution;
#[cfg(feature = "python")]
pub(crate) mod python_macros;
pub mod statistics;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule(name = "_stats_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use distribution::{
		Beta, BetaError, Exp, Gamma, GammaError, InverseGamma,
		InverseGammaError, Laplace, LaplaceError, LogNormal,
		LogNormalError, Normal, NormalError, Uniform, UniformError,
	};

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
