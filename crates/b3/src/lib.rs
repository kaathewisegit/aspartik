pub mod clock;
pub mod likelihood;
pub mod log;
pub mod mcmc;
pub mod operator;
pub mod parameter;
pub mod prior;
mod state;
pub mod substitution;
mod transitions;
mod tree;
pub mod util;

pub use log::PyLogger;
pub use prior::PyPrior;
pub use transitions::Transitions;
pub use tree::Tree;

use pyo3::prelude::*;

pub fn pymodule(py: Python) -> PyResult<Bound<PyModule>> {
	let m = PyModule::new(py, "_b3_rust_impl")?;

	m.add_submodule(&tree::submodule(py)?)?;

	m.add_class::<parameter::PyParameter>()?;
	m.add_class::<state::PyState>()?;
	m.add_class::<tree::PyTree>()?;
	m.add_class::<operator::Proposal>()?;
	m.add_class::<likelihood::PyLikelihood>()?;

	m.add_function(wrap_pyfunction!(mcmc::run, &m)?)?;

	Ok(m)
}
