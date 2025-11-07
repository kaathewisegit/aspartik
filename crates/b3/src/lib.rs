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

pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	use ::util::py_make_submodule;
	let m = py_make_submodule!(py, "_b3_rust_impl");

	m.add_class::<likelihood::PyCpu4Likelihood>()?;
	m.add_class::<likelihood::PyThread4Likelihood>()?;
	m.add_class::<likelihood::PyCudaLikelihood>()?;
	m.add_class::<mcmc::Mcmc>()?;
	m.add_class::<operator::PyProposal>()?;
	m.add_class::<tree::PyTree>()?;
	m.add_class::<tree::Leaf>()?;
	m.add_class::<tree::Internal>()?;

	m.add_class::<builtin::operators::EpochScale>()?;
	m.add_class::<builtin::operators::SubtreeLeap>()?;
	m.add_class::<builtin::priors::ConstantPopulation>()?;
	m.add_class::<builtin::priors::ExponentialGrowth>()?;
	m.add_class::<builtin::priors::Monophyly>()?;
	m.add_class::<builtin::priors::Yule>()?;

	m.add_class::<substitution::JC>()?;
	m.add_class::<substitution::K80>()?;
	m.add_class::<substitution::HKY>()?;

	Ok(m)
}
