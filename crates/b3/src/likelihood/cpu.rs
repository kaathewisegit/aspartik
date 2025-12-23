use anyhow::Result;
use data::{DnaNucleotide, Msa};
use linalg::Vector;
use num_traits::Inv;

use super::{LikelihoodTrait, Space};
use crate::{
	likelihood::{Linalg4, deduplicate},
	util::msa_to_likelihoods,
};
use skvec::{SkVec, skvec};

pub struct CpuLikelihood<S>
where
	S: Space,
{
	leaves: Vec<S::Vector>,
	projections: SkVec<S::Vector>,

	/// Pattern weights
	weights: Vec<S::Scalar>,

	num_sites: usize,
	num_leaves: usize,

	updated_edges: Vec<usize>,
	likelihoods: Vec<f64>,

	scales: SkVec<bool>,
	scale_sums: SkVec<u32>,

	scale: S::Scalar,
	inv_scale: S::Scalar,
	scale_ln: u32,
}

impl<S: Space> LikelihoodTrait for CpuLikelihood<S> {
	type S = S;

	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[S::Matrix],
		leaves_end: usize,
		root: usize,
		frequencies: S::Vector,
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
				let mut projection = transition * likelihood;

				let should_scale = if projection < self.scale {
					projection *= self.inv_scale;
					true
				} else {
					false
				};

				let projection_index = edge_idx + site;
				let old_scale = self.scales[projection_index];

				self.projections
					.set(projection_index, projection);

				if should_scale != old_scale {
					self.scales.set(
						projection_index,
						should_scale,
					);

					let old = self.scale_sums[site];

					let new = if should_scale {
						old + self.scale_ln
					} else {
						old - self.scale_ln
					};

					self.scale_sums.set(site, new);
				}
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
			let sum = S::sum(likelihood);
			let ln_sum = S::ln(sum);

			self.likelihoods[site] = ln_sum;
		}

		Ok(())
	}

	fn likelihood(&mut self) -> Result<f64> {
		let mut out: f64 = 0.0;

		for (likelihood, weight) in self
			.likelihoods
			.iter()
			.copied()
			.zip(self.weights.iter().copied())
		{
			let weight: f64 = weight.into();
			out += likelihood * weight;
		}

		for (scale_sum, weight) in self
			.scale_sums
			.iter()
			.copied()
			.zip(self.weights.iter().copied())
		{
			let scale_sum = f64::from(scale_sum);
			let weight: f64 = weight.into();
			out -= scale_sum * weight;
		}

		Ok(out)
	}

	fn accept(&mut self) -> Result<()> {
		self.projections.accept();
		self.scales.accept();
		self.scale_sums.accept();
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		let edges = std::mem::take(&mut self.updated_edges);
		let num_sites = self.num_sites;

		for edge in &edges {
			let edge_offset = edge * num_sites;
			for site in 0..num_sites {
				let index = edge_offset + site;
				self.projections.reject_element(index);
				self.scales.reject_element(index);
			}
		}

		// All of the edited items have been manually unset, so
		// there's no need for `accept` or `reject`.

		// small, so it's cheap to just reject
		self.scale_sums.reject();

		Ok(())
	}
}

impl CpuLikelihood<Linalg4> {
	pub fn new(msa: Msa<DnaNucleotide>, scale_ln: u32) -> Self {
		let (msa, weights) = deduplicate(msa);

		let num_sites = msa.num_sites();
		let num_leaves = msa.num_sequences();
		let num_internals = num_leaves - 1;
		let num_edges = num_internals * 2;

		let leaves = msa_to_likelihoods(msa);

		let projections =
			skvec![Vector::default(); num_edges * num_sites];
		let scales = skvec![false; num_edges * num_sites];
		let scale_sums = skvec![0; num_sites];

		let scale = (-<f64 as From<u32>>::from(scale_ln)).exp();

		Self {
			leaves,
			projections,

			num_sites,
			num_leaves,

			weights,

			updated_edges: Vec::new(),
			likelihoods: vec![f64::NAN; num_sites],

			scales,
			scale_sums,

			scale,
			inv_scale: scale.inv(),
			scale_ln,
		}
	}
}
