//! This crate aims is a fork of the [`statrs`][s] statistical utilities crate.
//! Currently it only provides distributions with support for sampling, a number
//! of statistics, and using all of those in Python.
//!
//! # Sampling
//!
//! The common use case is to set up the distributions and sample from them
//! which depends on the `Rand` crate for random number generation.
//!
#![cfg_attr(feature = "rand", doc = "```")]
#![cfg_attr(not(feature = "rand"), doc = "```ignore")]
//! use stats::distribution::Exp;
//! use rand::distr::Distribution;
//! use math::Positive;
//!
//! let mut r = rand::rng();
//! let n = Exp::new(Positive::new((0.5)));
//! print!("{}", n.sample(&mut r));
//! ```
//!
//! # Statistics
//!
//! ```
//! use stats::distribution::{
//!     Exp,
//!     // `cdf` and `pdf` methods
//!     Continuous, ContinuousCDF,
//! };
//! use stats::statistics::Distribution; // statistical moments and entropy
//! use math::Positive;
//!
//! let n = Exp::new(Positive::new((1.0)));
//! assert_eq!(n.mean(), Some(1.0));
//! assert_eq!(n.variance(), Some(1.0));
//! assert_eq!(n.entropy(), Some(1.0));
//! assert_eq!(n.skewness(), Some(2.0));
//! assert_eq!(n.cdf(1.0), 0.6321205588285577);
//! assert_eq!(n.pdf(1.0), 0.36787944117144233);
//! ```
//!
//! [s]: https://lib.rs/crates/statrs

#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
	use distribution::Beta;
	#[pymodule_export]
	use distribution::Exp;
	#[pymodule_export]
	use distribution::Gamma;
	#[pymodule_export]
	use distribution::InverseGamma;
	#[pymodule_export]
	use distribution::Laplace;
	#[pymodule_export]
	use distribution::LogNormal;
	#[pymodule_export]
	use distribution::Normal;
	#[pymodule_export]
	use distribution::Poisson;
	#[pymodule_export]
	use distribution::Uniform;

	#[pymodule_export]
	use distribution::BetaError;
	#[pymodule_export]
	use distribution::GammaError;
	#[pymodule_export]
	use distribution::InverseGammaError;
	#[pymodule_export]
	use distribution::LaplaceError;
	#[pymodule_export]
	use distribution::LogNormalError;
	#[pymodule_export]
	use distribution::NormalError;
	#[pymodule_export]
	use distribution::UniformError;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
