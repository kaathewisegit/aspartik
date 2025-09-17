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
//!
//! let mut r = rand::rng();
//! let n = Exp::new(0.5).unwrap();
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
//!
//! let n = Exp::new(1.0).unwrap();
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
pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	use util::py_make_submodule;
	let m = py_make_submodule!(py, "_stats_rust_impl");

	m.add_submodule(&distribution::pymodule(py)?)?;

	Ok(m)
}
