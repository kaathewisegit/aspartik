use anyhow::Result;
use pyo3::prelude::*;

use crate::parameters::{Node, PyReal, PyTree, Tree};

/// All nodes (leaves and internals) of a tree sorted by height
pub fn sorted_nodes(tree: &Tree) -> Vec<(Node, f64)> {
	let mut nodes = Vec::with_capacity(tree.num_nodes());

	for node in tree.nodes() {
		let height = tree.height_of(node);
		nodes.push((node, height));
	}

	// sort by height
	nodes.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

	nodes
}

trait Coalescent {
	type State: Copy;

	fn fetch_state(&self) -> Self::State;

	fn population_size_at(&self, point: f64, state: Self::State) -> f64;

	fn integral(&self, start: f64, end: f64, state: Self::State) -> f64;
}

fn calculate<C>(tree: &Tree, coalescent: &C) -> Result<f64>
where
	C: Coalescent,
{
	let state = coalescent.fetch_state();

	let nodes = sorted_nodes(tree);

	let mut out = 0.0; // log-likelihood
	let mut last_height = nodes[0].1;
	let mut num: usize = 1; // number of active lineages

	for (node, height) in nodes.into_iter().skip(1) {
		let binomial = (num * (num - 1) / 2) as f64;
		let area = coalescent.integral(last_height, height, state);
		out -= binomial * area;

		if tree.is_internal(node) {
			// merge event
			let pop = coalescent.population_size_at(height, state);
			out -= pop.ln();
			num -= 1;
		} else {
			// the node is a leaf, increase the number of linages
			num += 1;
		}

		last_height = height;
	}

	Ok(out)
}

/// Constant population coalescent
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct ConstantPopulation {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	population_size: Py<PyReal>,
}

impl Coalescent for ConstantPopulation {
	type State = f64;

	fn fetch_state(&self) -> f64 {
		self.population_size.get().inner().value()
	}

	fn population_size_at(&self, _point: f64, pop: f64) -> f64 {
		pop
	}

	fn integral(&self, start: f64, end: f64, pop: f64) -> f64 {
		(end - start) / pop
	}
}

#[pymethods]
impl ConstantPopulation {
	#[new]
	fn new(tree: Py<PyTree>, population_size: Py<PyReal>) -> Result<Self> {
		Ok(Self {
			tree,
			population_size,
		})
	}

	fn probability(&self) -> Result<f64> {
		let tree = self.tree.get().inner();
		calculate(&tree, self)
	}
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct ExponentialGrowth {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	population_size: Py<PyReal>,
	#[pyo3(get)]
	growth_rate: Py<PyReal>,
}

impl Coalescent for ExponentialGrowth {
	type State = (f64, f64);

	fn fetch_state(&self) -> Self::State {
		let pop = self.population_size.get().inner().value();
		let gr = self.growth_rate.get().inner().value();

		(pop, gr)
	}

	fn population_size_at(&self, point: f64, (pop, gr): (f64, f64)) -> f64 {
		pop * (-point * gr).exp()
	}

	fn integral(&self, start: f64, end: f64, (pop, gr): (f64, f64)) -> f64 {
		if gr == 0.0 {
			(end - start) / pop
		} else {
			((end * gr).exp() - (start * gr).exp()) / pop / gr
		}
	}
}

#[pymethods]
impl ExponentialGrowth {
	#[new]
	fn new(
		tree: Py<PyTree>,
		population_size: Py<PyReal>,
		growth_rate: Py<PyReal>,
	) -> Result<Self> {
		Ok(Self {
			tree,
			population_size,
			growth_rate,
		})
	}

	fn probability(&self) -> Result<f64> {
		let tree = self.tree.get().inner();
		calculate(&tree, self)
	}
}
