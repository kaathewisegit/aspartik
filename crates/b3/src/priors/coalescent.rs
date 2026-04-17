use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use crate::parameters::{Node, PyReal, PyTree, Tree};

#[derive(Debug)]
pub struct Intervals {
	state: Vec<(Node, f64)>,
	state_backup: Vec<(Node, f64)>,
}

impl Intervals {
	fn new(tree: &Tree) -> Mutex<Self> {
		let num_nodes = tree.num_nodes();
		let state = Vec::<(Node, f64)>::with_capacity(num_nodes);
		let mut out = Self {
			state_backup: Vec::new(),
			state,
		};
		out.rebuild(tree);
		out.state_backup.clone_from(&out.state);
		Mutex::new(out)
	}

	// Rebuilds `nodes` and `heights` from scratch
	fn rebuild(&mut self, tree: &Tree) {
		self.state.clear();

		for node in tree.nodes() {
			let height = tree.height_of(node);
			self.state.push((node, height));
		}

		self.state
			.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
	}

	fn update(&mut self, tree: &Tree) {
		let mut changed = Vec::new();
		for node in tree.internals() {
			if tree.is_node_height_updated(*node) {
				changed.push(node);
			}
			if changed.len() > 5 {
				self.rebuild(tree);
				return;
			}
		}

		for node in changed.iter().copied() {
			let idx = self
				.state
				.iter()
				.position(|(n, _)| *n == *node)
				// node must always be present in `nodes`
				.unwrap();

			self.state.remove(idx);
		}

		for node in changed {
			let height = tree.height_of(*node);

			let idx = self
				.state
				.binary_search_by(|probe| {
					probe.1.partial_cmp(&height).unwrap()
				})
				// both Ok and Err give us a valid insertion
				// point
				.unwrap_or_else(|e| e);
			self.state.insert(idx, (*node, height));
		}
	}

	fn accept(&mut self) {
		self.state_backup.copy_from_slice(&self.state);
	}

	fn reject(&mut self) {
		self.state.copy_from_slice(&self.state_backup);
	}
}

fn calculate<FP, FI>(
	intervals: &Intervals,
	tree: &Tree,
	population_size_at: FP,
	integral: FI,
) -> Result<f64>
where
	FP: Fn(f64) -> f64,
	FI: Fn(f64, f64) -> f64,
{
	let mut out = 0.0; // log-likelihood
	let mut last_height = intervals.state[0].1;
	let mut num_lineages = 1.0;

	for (node, height) in intervals.state.iter().skip(1) {
		let binomial = num_lineages * (num_lineages - 1.0) / 2.0;
		let area = integral(last_height, *height);
		out -= binomial * area;

		if tree.is_internal(*node) {
			// merge event
			let pop = population_size_at(*height);
			out -= pop.ln();
			num_lineages -= 1.0;
		} else {
			// the node is a leaf, increase the number of
			// linages
			num_lineages += 1.0;
		}

		last_height = *height;
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
	intervals: Mutex<Intervals>,
}

#[pymethods]
impl ConstantPopulation {
	#[new]
	fn new(tree: Py<PyTree>, population_size: Py<PyReal>) -> Result<Self> {
		let intervals = Intervals::new(&tree.get().inner());
		Ok(Self {
			tree,
			population_size,
			intervals,
		})
	}

	fn probability(&self) -> Result<f64> {
		let mut intervals = self.intervals.lock();
		let tree = self.tree.get().inner();
		let pop_size = self.population_size.get().inner().value();

		intervals.update(&tree);
		calculate(
			&intervals,
			&tree,
			|_point| pop_size,
			|start, end| (end - start) / pop_size,
		)
	}

	fn accept(&self) {
		self.intervals.lock().accept();
	}
	fn reject(&self) {
		self.intervals.lock().reject();
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
	intervals: Mutex<Intervals>,
}

#[pymethods]
impl ExponentialGrowth {
	#[new]
	fn new(
		tree: Py<PyTree>,
		population_size: Py<PyReal>,
		growth_rate: Py<PyReal>,
	) -> Result<Self> {
		let intervals = Intervals::new(&tree.get().inner());
		Ok(Self {
			tree,
			population_size,
			growth_rate,
			intervals,
		})
	}

	fn probability(&self) -> Result<f64> {
		let mut intervals = self.intervals.lock();
		let tree = self.tree.get().inner();
		let pop = self.population_size.get().inner().value();
		let gr = self.growth_rate.get().inner().value();

		intervals.update(&tree);

		calculate(
			&intervals,
			&tree,
			|point| pop * (-point * gr).exp(),
			|start, end| {
				if gr == 0.0 {
					(end - start) / pop
				} else {
					((end * gr).exp() - (start * gr).exp())
						/ pop / gr
				}
			},
		)
	}

	fn accept(&self) {
		self.intervals.lock().accept();
	}
	fn reject(&self) {
		self.intervals.lock().reject();
	}
}
