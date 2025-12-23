use core::f64;

use anyhow::Result;
use data::{DnaNucleotide, Msa};
use fork_union::{SyncMutPtr, ThreadPool};
use num_traits::{Float, Inv, NumCast, Zero};

use super::{LikelihoodTrait, Space};
use crate::{
	likelihood::{Linalg4, deduplicate},
	util::msa_to_likelihoods,
};
use linalg::Vector;

type Buffer<T> = Box<[T]>;

macro_rules! buffer {
	($element:expr; $length:expr) => {
		vec![$element; $length].into_boxed_slice()
	};
}

pub struct ParallelLikelihood<S>
where
	S: Space,
{
	pool: ThreadPool,

	projections: Buffer<S::Vector>,
	projections_backup: Buffer<S::Vector>,

	scales: Buffer<bool>,
	scales_backup: Buffer<bool>,
	scale_sums: Buffer<u32>,
	scale_sums_backup: Buffer<u32>,

	leaves: Vec<S::Vector>,

	weights: Vec<S::Scalar>,

	num_sites: usize,
	num_leaves: usize,

	updated_edges: Buffer<usize>,
	likelihoods: Buffer<S::Scalar>,

	scale_ln: u32,
	scale: S::Scalar,
	inv_scale: S::Scalar,
}

impl<S: Space> LikelihoodTrait for ParallelLikelihood<S> {
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

		self.updated_edges = edges.into();

		let num_sites = self.num_sites;
		let num_leaves = self.num_leaves;

		let projections =
			SyncMutPtr::new(self.projections.as_mut_ptr());
		let scales = SyncMutPtr::new(self.scales.as_mut_ptr());
		let scale_sums = SyncMutPtr::new(self.scale_sums.as_mut_ptr());

		for i in 0..leaves_end {
			let transition = transitions[i];

			let edge = edges[i];
			let edge_idx = edge * num_sites;

			let leaf = nodes[i];
			let leaf_idx = leaf * num_sites;

			self.pool.for_n(num_sites, |prong| {
				let site = prong.task_index;

				let leaf = self.leaves[leaf_idx + site];
				let projection = transition * leaf;

				let projection_index = edge_idx + site;

				unsafe {
					let ptr = projections
						.get(projection_index);
					ptr.write(projection);
				}
			});
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

			self.pool.for_n(num_sites, |prong| {
				let site = prong.task_index;

				let left = unsafe {
					projections.get(left_idx + site).read()
				};
				let right = unsafe {
					projections.get(right_idx + site).read()
				};

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
				unsafe {
					let projection_ptr = projections
						.get(projection_index);
					projection_ptr.write(projection);

					if should_scale != old_scale {
						let scale_ptr = scales
							.get(projection_index);
						scale_ptr.write(should_scale);

						let scale_sum_ptr =
							scale_sums.get(site);
						let old = scale_sum_ptr.read();

						let new = if should_scale {
							old + self.scale_ln
						} else {
							old - self.scale_ln
						};
						scale_sum_ptr.write(new);
					}
				}
			});
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
			let log_sum = S::sum(likelihood).ln();

			self.likelihoods[site] = log_sum;
		}

		Ok(())
	}

	fn likelihood(&mut self) -> Result<S::Scalar> {
		let mut out = S::Scalar::zero();

		for (likelihood, weight) in
			self.likelihoods.iter().zip(&self.weights)
		{
			out += *likelihood * *weight;
		}

		for (scale_sum, weight) in
			self.scale_sums.iter().zip(&self.weights)
		{
			let scale_sum =
				<S::Scalar as NumCast>::from(*scale_sum)
					.unwrap();
			out -= scale_sum * *weight;
		}

		Ok(out)
	}

	fn accept(&mut self) -> Result<()> {
		let num_sites = self.num_sites;

		for edge in &self.updated_edges {
			let start = edge * num_sites;
			let end = (edge + 1) * num_sites;
			self.projections_backup[start..end]
				.copy_from_slice(&self.projections[start..end]);
			self.scales_backup[start..end]
				.copy_from_slice(&self.scales[start..end]);
		}

		self.scale_sums_backup.copy_from_slice(&self.scale_sums);

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		let num_sites = self.num_sites;

		for edge in &self.updated_edges {
			let start = edge * num_sites;
			let end = (edge + 1) * num_sites;
			self.projections[start..end].copy_from_slice(
				&self.projections_backup[start..end],
			);
			self.scales[start..end].copy_from_slice(
				&self.scales_backup[start..end],
			);
		}

		self.scale_sums.copy_from_slice(&self.scale_sums_backup);

		Ok(())
	}
}

impl ParallelLikelihood<Linalg4> {
	pub fn new(
		msa: Msa<DnaNucleotide>,
		num_threads: usize,
		scale_ln: u32,
	) -> Result<Self> {
		let (msa, weights) = deduplicate(msa);

		let pool = ThreadPool::try_spawn(num_threads)?;

		let num_sites = msa.num_sites();
		let num_leaves = msa.num_sequences();
		let num_internals = num_leaves - 1;
		let num_edges = num_internals * 2;

		let leaves = msa_to_likelihoods(msa);

		let projections =
			buffer![Vector::default(); num_edges * num_sites];
		let scales = buffer![false; num_edges * num_sites];
		let scale_sums = buffer![0; num_sites];

		let scale = (-<f64 as From<u32>>::from(scale_ln)).exp();

		Ok(Self {
			pool,

			projections_backup: projections.clone(),
			projections,

			scales_backup: scales.clone(),
			scales,
			scale_sums_backup: scale_sums.clone(),
			scale_sums,

			leaves,

			weights,

			num_sites,
			num_leaves,

			updated_edges: Box::default(),
			likelihoods: buffer![f64::NAN; num_sites],

			scale_ln,
			scale,
			inv_scale: scale.inv(),
		})
	}
}
