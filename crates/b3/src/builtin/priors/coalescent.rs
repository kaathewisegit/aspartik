use anyhow::Result;
use pyo3::prelude::*;

use crate::tree::{Node, PyTree, Tree};

/// All nodes (leaves and internals) of a tree sorted by height
fn sorted_nodes(tree: &Tree) -> Vec<(Node, f64)> {
	let mut nodes = Vec::with_capacity(tree.num_nodes());

	for node in tree.nodes() {
		let height = tree.height_of(&node);
		nodes.push((node, height));
	}

	// sort by height
	nodes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

	nodes
}

trait Coalescent {
	fn population_size_at(&self, py: Python, point: f64) -> Result<f64>;
	fn integral(&self, py: Python, start: f64, end: f64) -> Result<f64>;
}

fn calculate<C>(py: Python, tree: &Tree, coalescent: &C) -> Result<f64>
where
	C: Coalescent,
{
	let nodes = sorted_nodes(tree);

	let mut out = 0.0; // log-likelihood
	let mut last_height = nodes[0].1;
	let mut num: usize = 1; // number of active lineages

	for (node, height) in nodes.into_iter().skip(1) {
		let binomial = (num * (num - 1) / 2) as f64;
		let area = coalescent.integral(py, last_height, height)?;
		out -= binomial * area;

		if tree.is_internal(&node) {
			// merge event
			let pop = coalescent.population_size_at(py, height)?;
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

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct ConstantPopulation {
	tree: Py<PyTree>,
	population: Py<PyAny>,
}

impl Coalescent for ConstantPopulation {
	fn population_size_at(&self, py: Python, _point: f64) -> Result<f64> {
		Ok(self.population_size(py).extract(py)?)
	}

	fn integral(&self, py: Python, start: f64, end: f64) -> Result<f64> {
		let pop: f64 = self.population_size(py).extract(py)?;
		Ok((end - start) / pop)
	}
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
	fn population_size(&self, py: Python) -> Py<PyAny> {
		self.population.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
		let tuple = (self.tree(py), self.population_size(py))
			.into_pyobject(py)?;

		Ok(tuple.into_any().unbind())
	}

	fn probability(&self, py: Python) -> Result<f64> {
		let tree = self.tree.get().inner();
		calculate(py, &tree, self)
	}
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct ExponentialGrowth {
	tree: Py<PyTree>,
	population_size: Py<PyAny>,
	growth_rate: Py<PyAny>,
}

impl Coalescent for ExponentialGrowth {
	fn population_size_at(&self, py: Python, point: f64) -> Result<f64> {
		let gr: f64 = self.growth_rate(py).extract(py)?;
		let pop: f64 = self.population_size(py).extract(py)?;

		let out = pop * (-point * gr).exp();
		Ok(out)
	}

	fn integral(&self, py: Python, start: f64, end: f64) -> Result<f64> {
		let gr: f64 = self.growth_rate(py).extract(py)?;
		let pop: f64 = self.population_size(py).extract(py)?;

		let out = if gr == 0.0 {
			(end - start) / pop
		} else {
			((end * gr).exp() - (start * gr).exp()) / pop / gr
		};

		Ok(out)
	}
}

#[pymethods]
impl ExponentialGrowth {
	#[new]
	fn new(
		tree: Py<PyTree>,
		population_size: Py<PyAny>,
		growth_rate: Py<PyAny>,
	) -> Self {
		// TODO: validation
		Self {
			tree,
			population_size,
			growth_rate,
		}
	}

	#[getter]
	fn tree(&self, py: Python) -> Py<PyTree> {
		self.tree.clone_ref(py)
	}

	#[getter]
	fn population_size(&self, py: Python) -> Py<PyAny> {
		self.population_size.clone_ref(py)
	}

	#[getter]
	fn growth_rate(&self, py: Python) -> Py<PyAny> {
		self.growth_rate.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
		let tuple = (
			self.tree(py),
			self.population_size(py),
			self.growth_rate(py),
		)
			.into_pyobject(py)?;

		Ok(tuple.into_any().unbind())
	}

	fn probability(&self, py: Python) -> Result<f64> {
		let tree = self.tree.get().inner();
		calculate(py, &tree, self)
	}
}
