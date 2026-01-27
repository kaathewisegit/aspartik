use std::collections::VecDeque;

use anyhow::Result;
use pyo3::{prelude::*, types::PyTuple};

use crate::parameters::{Internal, Leaf, PyTree, Tree};

/// Ensures that a group of leaves form a monophyly
///
/// Returns a static probability if the specified leaves are monophyletic or
/// aborts the move otherwise.
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct Monophyly {
	tree: Py<PyTree>,

	/// The leaves which must be [monophyletic][w]
	///
	/// [w]: https://en.wikipedia.org/wiki/Monophyly
	#[pyo3(get)]
	leaves: Vec<Leaf>,
}

#[pymethods]
impl Monophyly {
	#[new]
	fn new(tree: Py<PyTree>, leaves: Vec<Leaf>) -> Self {
		Self { tree, leaves }
	}

	#[getter]
	fn tree(&self, py: Python) -> Py<PyTree> {
		self.tree.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyTuple>> {
		(self.tree(py), self.leaves.clone())
			.into_pyobject(py)
			.map(|o| o.unbind())
	}

	fn probability(&self) -> Result<f64> {
		let tree = &*self.tree.get().inner();

		let mut mrca = tree
			.parent_of(&self.leaves[0])
			.expect("Leaves always have a parent");
		for leaf in self.leaves.iter().skip(1) {
			let parent = tree
				.parent_of(leaf)
				.expect("Leaves always have a parent");
			mrca = common_ancestor(tree, &mrca, &parent)?;
		}

		let mut count: usize = 0;
		let mut queue = VecDeque::from([mrca]);
		while let Some(internal) = queue.pop_front() {
			count += 1;
			let (left, right) = tree.children_of(&internal);
			if let Some(left) = tree.as_internal(&left) {
				queue.push_back(left);
			}
			if let Some(right) = tree.as_internal(&right) {
				queue.push_back(right);
			}
		}

		// the number of internals walked is higher than the number of a
		// minimal tree, meaning there are more leaves interleaved
		// between the passed ones
		if count != self.leaves.len() - 1 {
			return Ok(f64::NEG_INFINITY);
		}

		Ok(0.0)
	}
}

fn common_ancestor(
	tree: &Tree,
	a: &Internal,
	b: &Internal,
) -> Result<Internal> {
	let mut a = *a;
	let mut b = *b;

	while a != b {
		let height_a = tree.height_of(&a);
		let height_b = tree.height_of(&b);

		if height_a < height_b {
			a = tree.parent_of(&a).unwrap_or(a);
		} else if height_b < height_a {
			b = tree.parent_of(&b).unwrap_or(b);
		} else {
			// since branch lengths mustn't be 0, if `height_a ==
			// height_b`, neither of them is root.  This means we
			// can pick whichever.
			a = tree.parent_of(&a).unwrap_or(a);
		}
	}

	Ok(a)
}
