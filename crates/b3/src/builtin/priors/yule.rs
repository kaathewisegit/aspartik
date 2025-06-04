use anyhow::Result;
use pyo3::prelude::*;

use crate::tree::PyTree;

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct Yule {
	tree: Py<PyTree>,
	birth_rate: PyObject,
}

#[pymethods]
impl Yule {
	#[new]
	fn new(tree: Py<PyTree>, birth_rate: PyObject) -> Self {
		Self { tree, birth_rate }
	}

	fn probability(&self, py: Python) -> Result<f64> {
		let tree = self.tree.get().inner();
		let rate = self.birth_rate.bind(py).extract::<f64>()?;
		let root = tree.root();

		let mut out = (tree.num_leaves() - 1) as f64 * rate.ln();

		for internal in tree.internals() {
			let diff = -rate * tree.weight_of(internal.into());

			out += diff;
			if internal == root {
				out += diff;
			}
		}

		Ok(out)
	}
}
