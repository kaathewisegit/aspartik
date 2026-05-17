use anyhow::Result;
use pyo3::prelude::*;
use rand::seq::IndexedRandom;

use super::Proposal;
use crate::parameters::{Internal, Node, PyParameter, PyTree, Tree};
use rng::PyRng;
use util::{atomic::MonotonicF64, py_call_method};

/// Moves a node a distance, changing the topology randomly
///
/// First, a distance delta is sampled from the distribution.  The operator
/// selects a random node and all edges `delta` away from that node (down if the
/// delta is negative or up and down if it's positive).  One of those edges is
/// randomly selected and the node is spliced into it.  If the delta is above
/// the root, the node will become the new root.
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.operators", frozen)]
pub struct SubtreeLeap {
	/// The tree to edit.
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	/// The distribution to draw the height move distance from
	distribution: Py<PyAny>,
	#[pyo3(get)]
	rng: Py<PyRng>,
	#[pyo3(get)]
	weight: f64,

	factor: MonotonicF64,
}

#[pymethods]
impl SubtreeLeap {
	#[new]
	fn new(
		tree: Py<PyTree>,
		distribution: Py<PyAny>,
		rng: Py<PyRng>,
		weight: f64,
	) -> Result<Self> {
		Ok(Self {
			tree,
			distribution,
			rng,
			weight,

			factor: 0.75.into(),
		})
	}

	fn set_tuning(&self, parameter: f64) {
		self.factor.store(parameter);
	}

	fn get_tuning(&self) -> f64 {
		self.factor.load()
	}

	fn propose(&self, py: Python) -> Result<Proposal> {
		let tree = &mut *self.tree.get().inner();

		let delta: f64 = py_call_method!(
			py,
			self.distribution,
			"sample",
			self.rng.clone_ref(py)
		)?
		.extract(py)?;
		let delta = delta.abs() * (1.0 - self.factor.load());

		let rng = &mut *self.rng.get().inner();

		let (node, parent) = tree.random_nonroot_node(rng);
		let grandparent = tree.parent_of(*parent);
		let sibling = tree.other_child(parent, node)?;

		let destinations =
			walk_tree(tree, node, sibling, parent, delta)?;

		let (destination, destination_height) =
			*destinations.choose(rng).unwrap();
		let destination_parent = tree.parent_of(destination);

		if *parent == destination
			|| destination_parent.is_some_and(|d| d == parent)
		{
			// no topological changes
		} else {
			if let Some(grandparent) = grandparent {
				tree.replace_child(
					grandparent,
					*parent,
					sibling,
				)?;
			} else if let Some(sibling) = tree.as_internal(sibling)
			{
				tree.set_root(sibling);
			} else {
				unreachable!();
			}

			if let Some(destination_parent) = destination_parent {
				tree.replace_child(
					destination_parent,
					destination,
					*parent,
				)?;

				tree.replace_child(
					parent,
					sibling,
					destination,
				)?;
			} else {
				tree.replace_child(
					parent,
					sibling,
					destination,
				)?;
				tree.set_root(parent);
			}
		}

		tree.set_height(*parent, destination_height);

		let reverse_destinations = walk_tree(
			tree,
			node,
			tree.other_child(parent, node)?,
			parent,
			delta,
		)?;

		let num_dest = destinations.len() as f64;
		let rev_num_dest = reverse_destinations.len() as f64;

		Ok(Proposal::hastings(num_dest.ln() - rev_num_dest.ln()))
	}

	fn parameters(&self, py: Python) -> Vec<PyParameter> {
		vec![PyParameter::Tree(self.tree.clone_ref(py))]
	}
}

fn walk_tree(
	tree: &Tree,
	node: Node,
	sibling: Node,
	parent: Internal,
	delta: f64,
) -> Result<Vec<(Node, f64)>> {
	let mut destinations = Vec::<(Node, f64)>::new();

	let node_height = tree.height_of(node);
	let parent_height = tree.height_of(*parent);
	let (below, above) = (parent_height - delta, parent_height + delta);

	if node_height < below {
		intersections(&mut destinations, tree, sibling, below)
	}

	let mut up_node = parent;
	loop {
		let Some(up_parent) = tree.parent_of(*up_node) else {
			// up_node is root, terminate
			destinations.push((*up_node, above));
			break;
		};

		let up_parent_height = tree.height_of(*up_parent);

		if up_parent_height > above {
			// up_parent is above the line, `up_node` is a valid
			// destination
			destinations.push((*up_node, above));
			break;
		}

		// up_node is closer than delta

		let new_below = up_parent_height - (above - up_parent_height);

		let up_sibling = tree.other_child(up_parent, *up_node)?;

		if node_height < new_below {
			intersections(
				&mut destinations,
				tree,
				up_sibling,
				new_below,
			)
		}

		up_node = up_parent;
	}

	Ok(destinations)
}

fn intersections(
	destinations: &mut Vec<(Node, f64)>,
	tree: &Tree,
	node: Node,
	height: f64,
) {
	if tree.height_of(node) < height {
		destinations.push((node, height));
	} else if let Some(internal) = tree.as_internal(node) {
		let (left, right) = tree.children_of(internal);

		intersections(destinations, tree, left, height);
		intersections(destinations, tree, right, height);
	}
}
