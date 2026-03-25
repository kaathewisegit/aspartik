pub mod callback;
pub mod clock;
pub mod likelihood;
pub mod mcmc;
pub mod operators;
pub mod parameters;
pub mod priors;
pub mod substitution;
mod transitions;

pub use callback::PyCallback;
pub use transitions::Transitions;

use pyo3::prelude::*;

#[pymodule(name = "_b3_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use crate::{
		callback::TraceWriter,
		clock::PyClock,
		likelihood::{
			PyCpu4Likelihood, PyCudaLikelihood, PyHeteroLikelihood,
		},
		mcmc::Mcmc,
		operators::{
			ClassvecFlip, EpochScale, FixedHeightSPR, PyProposal,
			SubtreeLeap,
		},
		parameters::{
			Internal, Leaf, PyClassVector, PyReal, PyRealVector,
			PyTree,
		},
		priors::{
			ConstantPopulation, ExponentialGrowth, Monophyly,
			SymmetricDirichlet, Yule,
		},
		substitution::{PyGTR, PyHKY, PyJC, PyK80},
	};

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		util::py_patch_module!(m);

		Ok(())
	}
}
