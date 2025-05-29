#![expect(unused)]

use anyhow::Result;
use pyo3::prelude::*;

use crate::{parameter::PyParameter, tree::PyTree};

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct ConstantPopulation {
	tree: Py<PyTree>,
	population: Py<PyParameter>,
}

#[pymethods]
impl ConstantPopulation {
	#[new]
	fn new(tree: Py<PyTree>, population: Py<PyParameter>) -> Self {
		Self { tree, population }
	}

	fn probability(&self) -> Result<f64> {
		let tree = self.tree.get().inner();
		let pop = self.population.get().one_real()?;
		let mut nodes = Vec::with_capacity(tree.num_internals());

		for node in tree.internals() {
			let weight = tree.weight_of(node.into());
			nodes.push((node, weight));
		}
		nodes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

		let mut out = 1.0;
		let mut last_weight = 0.0;
		let mut num = tree.num_leaves();
		for (node, weight) in nodes {
			let time_diff = weight - last_weight;

			let binomial = num * (num - 1);
			let mult = binomial as f64 / pop;

			out += mult * (-mult * time_diff).exp();

			num -= 1;
			last_weight = weight;
		}

		Ok(out)
	}
}
