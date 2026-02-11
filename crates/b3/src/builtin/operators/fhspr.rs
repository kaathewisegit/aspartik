use anyhow::Result;
use pyo3::prelude::*;
use rand::seq::IteratorRandom;

use crate::{operator::Proposal, parameters::PyTree};
use rng::PyRng;

/// Fixed height subtree and regraft move.
///
/// This operator was described in [Hoehna et al 2008][l], section 3.2.7.  The
/// move selects a random node `i` and its parent `i_parent`.  It then selects a
/// random edge whose height overlaps with the height of `i_parent`.  `i_parent`
/// is spliced into the middle of this edge.
///
/// [l]: https://alexeidrummond.org/assets/publications/2008-hoehna-evalution.pdf
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.operators", frozen)]
pub struct FixedHeightSPR {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	rng: Py<PyRng>,
	#[pyo3(get)]
	weight: f64,
}

#[pymethods]
impl FixedHeightSPR {
	#[new]
	#[pyo3(signature = (tree, rng, weight = 1.0))]
	fn new(tree: Py<PyTree>, rng: Py<PyRng>, weight: f64) -> Result<Self> {
		Ok(Self { tree, rng, weight })
	}

	fn propose(&self) -> Result<Proposal> {
		let tree = &mut *self.tree.get().inner();
		let rng = &mut *self.rng.get().inner();

		let root = tree.root();

		let mut node = tree.random_node(rng);
		let mut parent = tree.parent_of(&node);
		while node == *root || parent.is_none_or(|p| p == root) {
			node = tree.random_node(rng);
			parent = tree.parent_of(&node);
		}
		let parent = parent.unwrap();

		let parent_height = tree.height_of(&parent);

		// random edge which intersects `parent_height`
		let edge = tree
			.edges()
			.filter(|edge| {
				let (node, parent) = tree.edge_nodes(*edge);
				tree.height_of(&node) < parent_height
					&& tree.height_of(&parent)
						> parent_height
			})
			.choose(rng);

		let Some(edge) = edge else {
			return Ok(Proposal::Reject());
		};

		let (other, _) = tree.edge_nodes(edge);

		tree.spr(&node, &other)?;

		Ok(Proposal::Hastings(0.0))
	}
}
