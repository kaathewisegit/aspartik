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
	use callback::TraceWriter;

	#[pymodule_export]
	use clock::PyClock;

	#[pymodule_export]
	use likelihood::{
		PyCpu4Likelihood, PyCudaLikelihood, PyHeteroLikelihood,
	};

	#[pymodule_export]
	use mcmc::Mcmc;
	#[pymodule_export]
	use operators::PyProposal;

	#[pymodule_export]
	use parameters::{
		Internal, Leaf, PyClassVector, PyReal, PyRealVector, PyTree,
	};

	#[pymodule_export]
	use operators::{
		ClassvecFlip, EpochScale, FixedHeightSPR, SubtreeLeap,
	};
	#[pymodule_export]
	use priors::{
		ConstantPopulation, ExponentialGrowth, Monophyly,
		SymmetricDirichlet, Yule,
	};

	#[pymodule_export]
	use substitution::{PyGTR, PyHKY, PyJC, PyK80};

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		util::py_patch_module!(m);

		Ok(())
	}
}
