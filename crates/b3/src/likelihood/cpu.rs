use anyhow::Result;
use data::{DnaNucleotide, Msa};

use super::{LikelihoodTrait, Row, Transition};
use crate::util::msa_to_likelihoods;
use skvec::{SkVec, skvec};

pub struct CpuLikelihood<const N: usize> {
	leaves: Vec<Row<N>>,
	projections: SkVec<Row<N>>,
	scales: SkVec<bool>,

	num_sites: usize,
	num_leaves: usize,

	updated_edges: Vec<usize>,
	likelihoods: Vec<f64>,
}

const SCALE: f64 = 1e-30;

impl LikelihoodTrait<4> for CpuLikelihood<4> {
	type Arguments = ();

	fn new(msa: Msa<DnaNucleotide>, _: ()) -> Result<Self> {
		let num_sites = msa.num_sites();
		let num_leaves = msa.num_sequences();
		let num_internals = num_leaves - 1;
		let num_edges = num_internals * 2;

		let leaves = msa_to_likelihoods(msa);

		let projections = skvec![Row::default(); num_edges * num_sites];
		let scales = skvec![false; num_edges];

		Ok(Self {
			leaves,
			projections,
			scales,

			num_sites,
			num_leaves,

			updated_edges: Vec::new(),
			likelihoods: vec![f64::NAN; num_sites],
		})
	}

	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<4>],
		leaves_end: usize,
		root: usize,
		frequencies: Row<4>,
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
		let num_sites = self.num_sites;

		let root_left_edge = (root - num_leaves) * 2;
		let root_right_edge = root_left_edge + 1;

		let root_left_idx = root_left_edge * num_sites;
		let root_right_idx = root_right_edge * num_sites;

		for site in 0..num_sites {
			let left = self.projections[root_left_idx + site];
			let right = self.projections[root_right_idx + site];
			let likelihood = left * right;
			let likelihood = likelihood * frequencies;
			let log_sum = likelihood.sum().ln();

			self.likelihoods[site] = log_sum;
		}

		Ok(())
	}

	fn likelihood(&mut self, weights: &[f64]) -> Result<f64> {
		let mut out = 0.0;

		for (likelihood, weight) in self.likelihoods.iter().zip(weights)
		{
			out += likelihood * weight;
		}

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
