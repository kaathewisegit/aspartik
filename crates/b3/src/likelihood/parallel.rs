use anyhow::Result;
use bytemuck::allocation::cast_vec;
use fork_union::{SyncMutPtr, ThreadPool};
use num_traits::{Float, Inv, Num, NumAssign};

use core::f64;
use std::ops::Mul;

use super::{
	LikelihoodTrait,
	cpu::{calc_leaf_projection, cast_transitions},
};
use linalg::{RowMatrix, Vector};

type Buffer<T> = Box<[T]>;

macro_rules! buffer {
	($element:expr; $length:expr) => {
		vec![$element; $length].into_boxed_slice()
	};
}

pub struct ParallelLikelihood<const N: usize, F> {
	internals_pool: ThreadPool,
	leaves_pool: ThreadPool,

	projections: Buffer<Vector<F, N>>,
	projections_backup: Buffer<Vector<F, N>>,

	scales: Buffer<bool>,
	scales_backup: Buffer<bool>,
	scale_sums: Buffer<u32>,
	scale_sums_backup: Buffer<u32>,

	leaves: Vec<u8>,

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
	f64: From<F> + From<u32>,
	RowMatrix<F, N, N>: Mul<Vector<F, N>, Output = Vector<F, N>>,
	Vector<F, N>: Mul<Output = Vector<F, N>>,
{
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[(usize, usize)],
		transitions: &[[[F; N]; N]],
		leaves_end: usize,
		frequencies: [F; N],
	) -> Result<()> {
		let transitions = cast_transitions(transitions);

		self.updated_edges = nodes.into();

		let num_sites = self.num_sites;
		let num_leaves = self.num_leaves;

		let projections =
			SyncMutPtr::new(self.projections.as_mut_ptr());
		let scales = SyncMutPtr::new(self.scales.as_mut_ptr());
		let scale_sums = SyncMutPtr::new(self.scale_sums.as_mut_ptr());

		let pool = if leaves_end > 10 {
			&mut self.leaves_pool
		} else {
			&mut self.internals_pool
		};

		pool.for_n(num_sites * leaves_end, |prong| {
			let i = prong.task_index / num_sites;
			let site = prong.task_index % num_sites;

			let transition = &transitions[i];

			let leaf = nodes[i];
			let leaf_idx = leaf * num_sites;

			let leaf = self.leaves[leaf_idx + site];
			let projection = calc_leaf_projection(transition, leaf);

			// SAFETY: for each iteration the destination is
			// thread-unique because `site`s are disjoint.
			unsafe {
				write_to(
					projections,
					leaf_idx + site,
					projection,
				);
			}
		});

		for i in leaves_end..nodes.len() - 1 {
			let transition = transitions[i];
			let node = nodes[i];

			let node_idx = node * num_sites;

			let (left_edge, right_edge) = children[i - leaves_end];

			let left_idx = left_edge * num_sites;
			let right_idx = right_edge * num_sites;

			self.internals_pool.for_n(num_sites, |prong| {
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

				let projection_index = node_idx + site;
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
					unsafe { scale_sum_ptr.write(new) }
				}
			});
		}

		let (_, _) = (scales, scale_sums);

		// XXX: deduplicate with cpu.rs?
		let root = nodes.last().unwrap();
		let (root_left_edge, root_right_edge) =
			children.last().unwrap();

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

			if self.scales[site] {
				self.scales[site] = false;
				self.scale_sums[site] -= self.scale_ln;
			}
		}

		Ok(())
	}

	fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()> {
		patterns.copy_from_slice(&self.likelihoods);

		for (i, scale_sum) in self.scale_sums.iter().enumerate() {
			patterns[i] -= f64::from(*scale_sum);
		}

		Ok(())
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
		num_sites: usize,
		leaves: Vec<u8>,
		num_leaf_threads: usize,
		num_internal_threads: usize,
		scale_ln: u32,
	) -> Result<Self> {
		let internals_pool =
			ThreadPool::try_spawn(num_internal_threads)?;
		let leaves_pool = ThreadPool::try_spawn(num_leaf_threads)?;

		let num_leaves = leaves.len() / num_sites;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let projections =
			buffer![Vector::default(); num_nodes * num_sites];
		let scales = buffer![false; num_nodes * num_sites];
		let scale_sums = buffer![0; num_sites];

		let scale = (-<f64 as From<u32>>::from(scale_ln)).exp();

		Ok(Self {
			internals_pool,
			leaves_pool,

			projections_backup: projections.clone(),
			projections,

			scales_backup: scales.clone(),
			scales,
			scale_sums_backup: scale_sums.clone(),
			scale_sums,

			leaves,

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
