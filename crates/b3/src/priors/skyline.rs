use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::coalescent::Intervals;
use crate::parameters::{Parameter, PyIntVector, PyRealVector, PyTree};

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct BayesianSkyline {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	pop_sizes: Py<PyRealVector>,
	#[pyo3(get)]
	group_sizes: Py<PyIntVector>,
	intervals: Mutex<Intervals>,
}

#[pymethods]
impl BayesianSkyline {
	#[new]
	fn new(
		tree: Py<PyTree>,
		pop_sizes: Py<PyRealVector>,
		group_sizes: Py<PyIntVector>,
	) -> Result<Self> {
		let intervals = Intervals::new(&tree.get().inner());

		let num_events = tree.get().inner().num_internals();
		let mut group_sizes_m = group_sizes.get().inner();
		let num_groups = group_sizes_m.len() as u32;

		for i in 0..num_groups {
			let mut size = num_events / num_groups;
			if i == num_groups - 1 {
				size += num_events % num_groups;
			}
			group_sizes_m.set(i as usize, size.into());
		}
		group_sizes_m.accept();

		drop(group_sizes_m);

		Ok(Self {
			tree,
			pop_sizes,
			group_sizes,
			intervals,
		})
	}

	fn probability(&self) -> Result<f64> {
		let intervals = &mut *self.intervals.lock();
		let tree = &*self.tree.get().inner();
		let group_sizes = &*self.group_sizes.get().inner();
		let pop_sizes = &*self.pop_sizes.get().inner();

		intervals.update(tree);

		let mut group_index = 0;
		let mut sub_index = 0;

		let mut pop_size;

		let mut out = 0.0; // log-likelihood
		let mut last_height = intervals.state()[0].1;
		let mut num_lineages = 1.0;

		// duplicated from coalescent::calculate
		for (node, height) in intervals.state().iter().skip(1) {
			pop_size = pop_sizes[group_index];
			if tree.is_internal(*node) {
				sub_index += 1;
				if sub_index
					>= group_sizes[group_index] as usize
				{
					group_index += 1;
					sub_index = 0;
				}
			}

			let binomial =
				num_lineages * (num_lineages - 1.0) / 2.0;
			let area = (*height - last_height) / pop_size;
			out -= binomial * area;

			if tree.is_internal(*node) {
				out -= pop_size.ln();
				num_lineages -= 1.0;
			} else {
				num_lineages += 1.0;
			}

			last_height = *height;
		}

		Ok(out)
	}

	fn is_changed(&self) -> bool {
		self.tree.get().is_changed()
			|| self.pop_sizes.get().is_changed()
			|| self.group_sizes.get().is_changed()
	}

	fn accept(&self) {
		self.intervals.lock().accept();
	}

	fn reject(&self) {
		self.intervals.lock().reject();
	}
}
