mod builtin;
pub mod callback;
pub mod clock;
pub mod likelihood;
pub mod mcmc;
pub mod operator;
pub mod parameter;
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

	m.add_submodule(&tree::submodule(py)?)?;

	m.add_class::<likelihood::PyLikelihood>()?;
	m.add_class::<mcmc::Mcmc>()?;
	m.add_class::<operator::Proposal>()?;
	m.add_class::<parameter::PyBoolean>()?;
	m.add_class::<parameter::PyInteger>()?;
	m.add_class::<parameter::PyReal>()?;
	m.add_class::<tree::PyTree>()?;

	m.add_class::<builtin::operators::EpochScale>()?;
	m.add_class::<builtin::operators::SubtreeLeap>()?;
	m.add_class::<builtin::priors::ConstantPopulation>()?;
	m.add_class::<builtin::priors::ExponentialGrowth>()?;
	m.add_class::<builtin::priors::Monophyly>()?;
	m.add_class::<builtin::priors::Yule>()?;

	Ok(m)
}
