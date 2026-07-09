use std::collections::VecDeque;

use anyhow::Result;
use pyo3::prelude::*;

use crate::parameters::{Leaf, PyTree};

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct Monophyly {
	#[pyo3(get)]
	tree: Py<PyTree>,

	/// The leaves should must be [monophyletic][w]
	///
	/// [w]: https://en.wikipedia.org/wiki/Monophyly
	#[pyo3(get)]
	leaves: Vec<Leaf>,

	/// Log-probability penalty to apply if the leaves aren't monophyletic
	penalty: f64,
}

#[pymethods]
impl Monophyly {
	#[new]
	#[pyo3(signature = (tree, leaves, penalty = -1000.0))]
	fn new(tree: Py<PyTree>, leaves: Vec<Leaf>, penalty: f64) -> Self {
		Self {
			tree,
			leaves,
			penalty,
		}
	}

	fn probability(&self) -> Result<f64> {
		let tree = &*self.tree.get().inner();
		let num_leaves = self.leaves.len();

		let mut mrca = tree.mrca(*self.leaves[0], *self.leaves[1]);
		for leaf in self.leaves.iter().skip(2).copied() {
			mrca = tree.mrca(*mrca, *leaf);
		}

		let mut count: usize = 0;
		let mut queue = VecDeque::from([mrca]);
		while let Some(internal) = queue.pop_front() {
			count += 1;
			let (left, right) = tree.children_of(internal);
			if let Some(left) = tree.as_internal(left) {
				queue.push_back(left);
			}
			if let Some(right) = tree.as_internal(right) {
				queue.push_back(right);
			}

			if count >= num_leaves {
				return Ok(self.penalty);
			}
		}

		Ok(0.0)
	}

	fn is_changed(&self) -> bool {
		self.tree.get().is_changed()
	}
}
