mod builtin;
pub mod callback;
pub mod clock;
pub mod likelihood;
pub mod mcmc;
pub mod operator;
pub mod prior;
pub mod substitution;
mod transitions;
mod tree;
pub mod util;

pub use callback::PyCallback;
pub use prior::PyPrior;
pub use transitions::Transitions;
pub use tree::Tree;

use pyo3::prelude::*;

#[pymodule(name = "_b3_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use likelihood::PyCpu4Likelihood;
	#[pymodule_export]
	use likelihood::PyCudaLikelihood;
	#[pymodule_export]
	use likelihood::PyParallel4Likelihood;

	#[pymodule_export]
	use mcmc::Mcmc;
	#[pymodule_export]
	use operator::PyProposal;
	#[pymodule_export]
	use tree::Internal;
	#[pymodule_export]
	use tree::Leaf;
	#[pymodule_export]
	use tree::PyTree;

	#[pymodule_export]
	use builtin::operators::EpochScale;
	#[pymodule_export]
	use builtin::operators::SubtreeLeap;
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
