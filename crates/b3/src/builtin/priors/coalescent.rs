use anyhow::Result;
use pyo3::prelude::*;

use crate::tree::PyTree;

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct ConstantPopulation {
	tree: Py<PyTree>,
	population: Py<PyAny>,
}

#[pymethods]
impl ConstantPopulation {
	#[new]
	fn new(tree: Py<PyTree>, population: Py<PyAny>) -> Self {
		Self { tree, population }
	}

	#[getter]
	fn tree(&self, py: Python) -> Py<PyTree> {
		self.tree.clone_ref(py)
	}

	#[getter]
	fn population(&self, py: Python) -> Py<PyAny> {
		self.population.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
		let tuple = (self.tree(py), self.population(py))
			.into_pyobject(py)?;

		Ok(tuple.into_any().unbind())
	}

	fn probability(&self, py: Python) -> Result<f64> {
		let tree = self.tree.get().inner();
		let pop = self.population.bind(py).extract::<f64>()?;
		let mut nodes = Vec::with_capacity(tree.num_nodes());

		for node in tree.nodes() {
			let height = tree.height_of(&node);
			nodes.push((node, height));
		}
		// sort by height
		nodes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

		let mut out = 1.0;
		let mut last_height = 0.0;
		let mut num: usize = 0;

		for (node, height) in nodes {
			if tree.is_leaf(&node) {
				// no transition event.  Increase the num for
				// the new number of lineages which could merge
				num += 1;
				continue;
			}

			// the node is internal, merge event

			let time_diff = height - last_height;

			let binomial = num * (num - 1);
			let mult = binomial as f64 / pop;

			out += mult * (-mult * time_diff).exp();

			num -= 1;
			last_height = height;
		}

		Ok(out)
	}
}
