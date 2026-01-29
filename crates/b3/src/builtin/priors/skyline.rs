#![allow(unused)]

use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::coalescent::sorted_nodes;
use crate::parameters::{PyTree, Tree};

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct Skyline {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	mutation_rate: f64,
	#[pyo3(get)]
	epsilon: f64,
	population_sizes: Mutex<Vec<f64>>,
}

#[pymethods]
impl Skyline {
	#[new]
	fn new(
		tree: Py<PyTree>,
		mutation_rate: f64,
		epsilon: f64,
	) -> Result<Self> {
		let num_coalescents = tree.get().num_internals();
		// TODO: init + refresh
		let population_sizes = vec![0.0; num_coalescents].into();

		Ok(Self {
			tree,
			mutation_rate,
			epsilon,

			population_sizes,
		})
	}

	fn probability(&self, _py: Python) -> Result<f64> {
		let mut out = 0.0;

		let tree = self.tree.get().inner();
		let nodes = sorted_nodes(&tree);
		let mut num_lineages = tree.num_leaves();

		let population_sizes = self.population_sizes.lock();

		for i in 0..nodes.len() {
			let segment_len = nodes[i].1;

			let binom = ((num_lineages * (num_lineages - 1)) / 2)
				as f64;

			if tree.is_internal(&nodes[i].0) {
				// coalescent event
				out += (binom / population_sizes[i]).ln();
				num_lineages -= 1;
			}
			out -= segment_len * binom / population_sizes[i];
		}

		Ok(out)
	}
}
