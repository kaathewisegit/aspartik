use anyhow::Result;

use super::{LikelihoodTrait, Row, Transition};
use skvec::{skvec, SkVec};

pub struct CpuLikelihood<const N: usize> {
	leaves: Vec<Vec<Row<N>>>,
	projections: SkVec<Row<N>>,

	updated_edges: Vec<usize>,
}

impl<const N: usize> LikelihoodTrait<N> for CpuLikelihood<N> {
	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<N>],
		cuttoff: usize,
		root: usize,
	) -> Result<f64> {
		assert_eq!(nodes.len(), edges.len());
		assert_eq!(nodes.len(), transitions.len());

		self.updated_edges = edges.to_vec();

		let num_sites = self.num_sites();
		let num_leaves = self.num_leaves();
		let num_edges = self.num_edges();

		for site in 0..num_sites {
			let offset = site * num_edges;

			for i in 0..cuttoff {
				let projection = transitions[i]
					* self.leaves[site][nodes[i]];

				self.projections
					.set(offset + edges[i], projection);
			}
		}

		for site in 0..num_sites {
			let offset = site * num_edges;

			for i in cuttoff..nodes.len() {
				let left_idx =
					offset + (nodes[i] - num_leaves) * 2;
				let right_idx = left_idx + 1;
				let left = self.projections[left_idx];
				let right = self.projections[right_idx];

				let likelihood = left * right;
				let projection = transitions[i] * likelihood;

				self.projections
					.set(offset + edges[i], projection);
			}
		}

		let mut out = 0.0;
		let root_left = (root - num_leaves) * 2;
		let root_right = root_left + 1;
		for i in 0..num_sites {
			let offset = i * num_edges;

			let left = self.projections[offset + root_left];
			let right = self.projections[offset + root_right];
			let likelihood = left * right;
			let log_sum = likelihood.sum().ln();
			out += log_sum;
		}

		Ok(out)
	}

	fn accept(&mut self) -> Result<()> {
		self.projections.accept();
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		let edges = std::mem::take(&mut self.updated_edges);

		let num_edges = self.num_edges();
		for i in 0..self.num_sites() {
			let offset = i * num_edges;
			for edge in &edges {
				self.projections.reject_element(*edge + offset);
			}
		}
		// All of the edited items have been manually unset, so
		// there's no need for `accept` or `reject`.

		Ok(())
	}
}

impl<const N: usize> CpuLikelihood<N> {
	pub fn new(leaves: Vec<Vec<Row<N>>>) -> Self {
		let num_sites = leaves.len();
		let num_leaves = leaves[0].len();
		let num_internals = num_leaves - 1;
		let num_edges = num_internals * 2;

		let projections = skvec![Row::default(); num_edges * num_sites];

		Self {
			leaves,
			projections,

			updated_edges: Vec::new(),
		}
	}

	fn num_sites(&self) -> usize {
		self.leaves.len()
	}

	fn num_leaves(&self) -> usize {
		self.leaves[0].len()
	}

	fn num_edges(&self) -> usize {
		(self.num_leaves() - 1) * 2
	}
}
