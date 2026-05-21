use anyhow::Result;
use pyo3::prelude::*;

use crate::parameters::{PyReal, PyTree};

/// Uncalibrated Yule birth-rate model
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct Yule {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	birth_rate: Py<PyReal>,
}

#[pymethods]
impl Yule {
	#[new]
	fn new(tree: Py<PyTree>, birth_rate: Py<PyReal>) -> Self {
		Self { tree, birth_rate }
	}

	fn probability(&self) -> Result<f64> {
		let tree = self.tree.get().inner();
		let rate = self.birth_rate.get().inner().value();
		let root = tree.root();

		let mut out = tree.num_internals() as f64 * rate.ln();

		for internal in tree.internals() {
			let diff = -rate * tree.height_of(*internal);

			out += diff;
			if internal == root {
				out += diff;
			}
		}

		Ok(out)
	}

	fn is_changed(&self) -> bool {
		self.tree.get().is_changed()
			|| self.birth_rate.get().is_changed()
	}
}
