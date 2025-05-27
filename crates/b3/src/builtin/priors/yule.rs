use anyhow::Result;
use pyo3::prelude::*;

use crate::{parameter::PyParameter, tree::PyTree};

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct Yule {
	tree: Py<PyTree>,
	birth_rate: Py<PyParameter>,
}

#[pymethods]
impl Yule {
	#[new]
	fn new(tree: Py<PyTree>, birth_rate: Py<PyParameter>) -> Self {
		Self { tree, birth_rate }
	}

	fn probability(&self) -> Result<f64> {
		let tree = self.tree.get().inner();
		let rate = self.birth_rate.get().one_real()?;
		let log_rate = rate.ln();
		let root = tree.root();

		let mut out = (tree.num_leaves() - 1) as f64 * log_rate;

		for internal in tree.internals() {
			let diff = -log_rate * tree.weight_of(internal.into());

			out += diff;
			if internal == root {
				out += diff;
			}
		}

		Ok(out)
	}
}
