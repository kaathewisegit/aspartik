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

use rng::PyRng;

#[pyfunction]
fn test(_rng: PyRng) {}

/// Short title.
///
/// Description.
#[pymodule(name = "_b3_rust_impl")]
pub fn pymodule(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
	let tree = tree::submodule(py)?;
	m.add_submodule(&tree)?;
	// py.import("sys")?
	// 	.getattr("modules")?
	// 	.set_item("b3.tree", tree)?;

	m.add_class::<parameter::PyParameter>()?;
	m.add_class::<state::PyState>()?;
	m.add_class::<tree::PyTree>()?;
	m.add_class::<operator::Proposal>()?;
	m.add_class::<likelihood::PyLikelihood>()?;

	m.add_function(wrap_pyfunction!(mcmc::run, m)?)?;
	m.add_function(wrap_pyfunction!(test, m)?)?;

	Ok(())
}
