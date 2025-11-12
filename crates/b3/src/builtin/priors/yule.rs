use anyhow::Result;
use pyo3::prelude::*;

use crate::tree::PyTree;
use pyutil::SupportsFloat;

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct Yule {
	tree: Py<PyTree>,
	birth_rate: SupportsFloat,
}

#[pymethods]
impl Yule {
	#[new]
	fn new(tree: Py<PyTree>, birth_rate: SupportsFloat) -> Self {
		Self { tree, birth_rate }
	}

	#[getter]
	fn tree(&self, py: Python) -> Py<PyTree> {
		self.tree.clone_ref(py)
	}

	#[getter]
	fn birth_rate(&self, py: Python) -> SupportsFloat {
		self.birth_rate.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
		let tuple = (self.tree(py), self.birth_rate(py))
			.into_pyobject(py)?;

		Ok(tuple.into_any().unbind())
	}

	fn probability(&self, py: Python) -> Result<f64> {
		let tree = self.tree.get().inner();
		let rate = self.birth_rate.extract(py)?;
		let root = tree.root();

		let mut out = (tree.num_internals() - 1) as f64 * rate.ln();

		for internal in tree.internals() {
			let diff = -rate * tree.height_of(&internal);

			out += diff;
			if internal == root {
				out += diff;
			}
		}

		Ok(out)
	}
}
