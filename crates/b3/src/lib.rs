mod builtin;
pub mod callback;
pub mod clock;
pub mod likelihood;
pub mod mcmc;
pub mod operator;
pub mod parameters;
pub mod prior;
pub mod substitution;
mod transitions;
pub mod util;

pub use callback::PyCallback;
pub use prior::PyPrior;
pub use transitions::Transitions;

use pyo3::prelude::*;

#[pymodule(name = "_b3_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use clock::PyClock;

	#[pymodule_export]
	use likelihood::{
		PyCpu4Likelihood, PyCudaLikelihood, PyHeteroLikelihood,
		PyParallel4Likelihood,
	};

	#[pymodule_export]
	use mcmc::Mcmc;
	#[pymodule_export]
	use operator::PyProposal;

	#[pymodule_export]
	use parameters::{Internal, Leaf, PyClassVector, PyRealVector, PyTree};

	#[pymodule_export]
	use parameters::PyReal;

	#[pymodule_export]
	use builtin::operators::{ClassvecFlip, EpochScale, SubtreeLeap};
	#[pymodule_export]
	use builtin::priors::ConstantPopulation;
	#[pymodule_export]
	use builtin::priors::ExponentialGrowth;
	#[pymodule_export]
	use builtin::priors::Monophyly;
	#[pymodule_export]
	use builtin::priors::Yule;

	#[pymodule_export]
	use substitution::HKY;
	#[pymodule_export]
	use substitution::JC;
	#[pymodule_export]
	use substitution::K80;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
