use anyhow::Result;

use super::{LikelihoodTrait, Row, Transition};
use crate::util::transpose;
use skvec::{skvec, SkVec};

pub struct CpuLikelihood<const N: usize> {
	leaves: Vec<Row<N>>,
	projections: SkVec<Row<N>>,
	scales: SkVec<bool>,

	num_sites: usize,
	num_leaves: usize,

	updated_edges: Vec<usize>,

	likelihood: f64,
}

const SCALE: f64 = 1e-30;

impl<const N: usize> LikelihoodTrait<N> for CpuLikelihood<N> {
	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<N>],
		leaves_end: usize,
		root: usize,
	) -> Result<()> {
		assert_eq!(nodes.len(), edges.len());
		assert_eq!(nodes.len(), transitions.len());

		self.updated_edges = edges.to_vec();

		let num_sites = self.num_sites;
		let num_leaves = self.num_leaves;

		for i in 0..leaves_end {
			let transition = transitions[i];

			let edge = edges[i];
			let edge_idx = edge * num_sites;

			let leaf = nodes[i];
			let leaf_idx = leaf * num_sites;

			for site in 0..num_sites {
				let leaf = self.leaves[leaf_idx + site];
				let projection = transition * leaf;

				self.projections
					.set(edge_idx + site, projection);
			}
		}

		for i in leaves_end..nodes.len() {
			let mut should_scale = true;

			let transition = transitions[i];
			let node = nodes[i];

			let edge = edges[i];
			let edge_idx = edge * num_sites;

			let left_edge = (node - num_leaves) * 2;
			let right_edge = left_edge + 1;

			let left_idx = left_edge * num_sites;
			let right_idx = right_edge * num_sites;

			for site in 0..num_sites {
				let left = self.projections[left_idx + site];
				let right = self.projections[right_idx + site];

				let likelihood = left * right;
				let projection = transition * likelihood;
				should_scale &= projection[0] < SCALE;
				should_scale &= projection[1] < SCALE;
				should_scale &= projection[2] < SCALE;
				should_scale &= projection[3] < SCALE;

				self.projections
					.set(edge_idx + site, projection);
			}

			if should_scale {
				for site in 0..num_sites {
					let mut projection = self.projections
						[edge_idx + site];
					projection /= SCALE;
					self.projections.set(
						edge_idx + site,
						projection,
					);
				}
				self.scales.set(edge, true);
			} else {
				self.scales.set(edge, false);
			}
		}

		let num_leaves = self.num_leaves;

		let mut out_likelihood = 0.0;

		let root_left_edge = (root - num_leaves) * 2;
		let root_right_edge = root_left_edge + 1;

		let root_left_idx = root_left_edge * num_sites;
		let root_right_idx = root_right_edge * num_sites;

		for site in 0..num_sites {
			let left = self.projections[root_left_idx + site];
			let right = self.projections[root_right_idx + site];
			let likelihood = left * right;
			let log_sum = likelihood.sum().ln();
			out_likelihood += log_sum;
		}

		self.likelihood = out_likelihood;

		Ok(())
	}

	fn likelihood(&mut self) -> Result<f64> {
		let mut out = self.likelihood;

		for scaled in &self.scales {
			if *scaled {
				out += SCALE.ln() * self.num_sites as f64;
			}
		}

		Ok(out)
	}

	fn accept(&mut self) -> Result<()> {
		self.projections.accept();
		self.scales.accept();
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		let edges = std::mem::take(&mut self.updated_edges);
		let num_sites = self.num_sites;

		for edge in &edges {
			let edge_offset = edge * num_sites;
			for site in 0..num_sites {
				self.projections
					.reject_element(edge_offset + site);
			}
		}

		// All of the edited items have been manually unset, so
		// there's no need for `accept` or `reject`.

		// small, so it's cheap to just reject
		self.scales.reject();

		Ok(())
	}
}

impl<const N: usize> CpuLikelihood<N> {
	pub fn new(leaves: Vec<Vec<Row<N>>>) -> Self {
		let num_sites = leaves.len();
		let num_leaves = leaves[0].len();
		let num_internals = num_leaves - 1;
		let num_edges = num_internals * 2;

		let leaves = transpose(leaves);

		let projections = skvec![Row::default(); num_edges * num_sites];
		let scales = skvec![false; num_edges];

		Self {
			leaves,
			projections,
			scales,

			num_sites,
			num_leaves,

			updated_edges: Vec::new(),

			likelihood: f64::NAN,
		}
	}
}
