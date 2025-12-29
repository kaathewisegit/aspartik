use anyhow::Result;
use data::{DnaNucleotide, Msa};
use fork_union::{SyncMutPtr, ThreadPool};
use num_traits::Inv;
use num_traits::{Float, Num, NumAssign};

use core::f64;
use std::ops::Mul;

use super::LikelihoodTrait;
use crate::{likelihood::deduplicate, util::msa_to_likelihoods};
use linalg::{RowMatrix, Vector};

type Buffer<T> = Box<[T]>;

macro_rules! buffer {
	($element:expr; $length:expr) => {
		vec![$element; $length].into_boxed_slice()
	};
}

pub struct ParallelLikelihood<const N: usize, F> {
	pool: ThreadPool,

	projections: Buffer<Vector<F, N>>,
	projections_backup: Buffer<Vector<F, N>>,

	scales: Buffer<bool>,
	scales_backup: Buffer<bool>,
	scale_sums: Buffer<u32>,
	scale_sums_backup: Buffer<u32>,

	leaves: Vec<Vector<F, N>>,

	weights: Vec<F>,

	num_sites: usize,
	num_leaves: usize,

	updated_edges: Buffer<usize>,
	likelihoods: Buffer<f64>,

	scale_ln: u32,
	scale: F,
	inv_scale: F,
}

/// Write `value` to the `index` position of `sync_ptr`
///
/// # Safety
///
/// - Index must be within bounds of the `sync_ptr` allocation
/// - No other thread must be reading or writing to the position at `index`
unsafe fn write_to<T>(sync_ptr: SyncMutPtr<T>, index: usize, value: T) {
	let ptr = unsafe { sync_ptr.get(index) };
	unsafe { ptr.write(value) };
}

impl<const N: usize, F> LikelihoodTrait<N, F> for ParallelLikelihood<N, F>
where
	F: Float + Num + NumAssign + Send + Sync,
	f64: From<F>,
	RowMatrix<F, N, N>: Mul<Vector<F, N>, Output = Vector<F, N>>,
	Vector<F, N>: Mul<Output = Vector<F, N>>,
{
	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[[[F; N]; N]],
		leaves_end: usize,
		root: usize,
		frequencies: [F; N],
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
			let transition = RowMatrix::from(transitions[i]);

			let edge = edges[i];
			let edge_idx = edge * num_sites;

			let leaf = nodes[i];
			let leaf_idx = leaf * num_sites;

			self.pool.for_n(num_sites, |prong| {
				let site = prong.task_index;

				let leaf = self.leaves[leaf_idx + site];
				let projection = transition * leaf;

				let projection_index = edge_idx + site;

				// SAFETY: for each iteration `projection_index`
				// is thread-unique because `site`s are
				// disjoint.
				unsafe {
					write_to(
						projections,
						projection_index,
						projection,
					);
				}
			});
		}

		for i in leaves_end..nodes.len() {
			let transition = RowMatrix::from(transitions[i]);
			let node = nodes[i];

			let edge = edges[i];
			let edge_idx = edge * num_sites;

			let left_edge = (node - num_leaves) * 2;
			let right_edge = left_edge + 1;

			let left_idx = left_edge * num_sites;
			let right_idx = right_edge * num_sites;

			self.pool.for_n(num_sites, |prong| {
				let site = prong.task_index;

				// SAFETY: `site` is unique to us, the code in
				// the closure is serial
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

				// SAFETY: `projection_index` is
				// disjoint on `site`, see `unsafe` in
				// the leaf calculations.
				unsafe {
					write_to(
						projections,
						projection_index,
						projection,
					);
				}

				if should_scale != old_scale {
					// SAFETY: see above
					unsafe {
						write_to(
							scales,
							projection_index,
							should_scale,
						);
					}

					let scale_sum_ptr =
						// SAFETY: `site` is disjoint
						unsafe { scale_sums.get(site) };
					let old =
						// SAFETY: we are the only
						// thread which is reading from
						// this pointer right now
						unsafe { scale_sum_ptr.read() };

					let new = if should_scale {
						old + self.scale_ln
					} else {
						old - self.scale_ln
					};

					// SAFETY: `site` is disjoint
					// between threads
					unsafe { scale_sum_ptr.write(new) }
				}
			});
		}

		let num_leaves = self.num_leaves;
		let num_sites = self.num_sites;

		let root_left_edge = (root - num_leaves) * 2;
		let root_right_edge = root_left_edge + 1;

		let root_left_idx = root_left_edge * num_sites;
		let root_right_idx = root_right_edge * num_sites;

		let frequencies = Vector::from(frequencies);
		for site in 0..num_sites {
			let left = self.projections[root_left_idx + site];
			let right = self.projections[root_right_idx + site];
			let likelihood = left * right;
			let likelihood = likelihood * frequencies;
			let ln_sum = likelihood.sum().ln();

			self.likelihoods[site] = ln_sum.into();
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
			let scale_sum: f64 = scale_sum.into();
			let weight: f64 = weight.into();
			out -= scale_sum * weight;
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

impl ParallelLikelihood<4, f64> {
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
