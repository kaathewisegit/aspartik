use anyhow::Result;
use num_traits::Inv;

use std::{
	ops::Range,
	slice::{from_raw_parts, from_raw_parts_mut},
};

use super::Calculator;

pub struct CpuLikelihood<const N: usize, F> {
	num_patterns: usize,
	num_leaves: usize,

	selectors: Vec<u8>,
	selectors_backup: Vec<u8>,

	/// Have the length of `num_patterns`
	samples: Vec<u8>,
	projections: Vec<[F; N]>,

	likelihoods: Vec<f64>,

	scales: Vec<bool>,
	scale_sums: Vec<u32>,
	scale_sums_backup: Vec<u32>,

	/// Scaling threshold on logarithmic scale
	scale_ln: u32,
	/// `e^(-scale_ln)`
	scale_threshold: F,
	/// `e^scale_ln`
	scale_mult: F,
}

impl<const N: usize> Calculator<N, f64> for CpuLikelihood<N, f64> {
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[(usize, usize)],
		transitions: &[[[f64; N]; N]],
		leaves_end: usize,
		frequencies: [f64; N],
	) -> Result<()> {
		let num_patterns = self.num_patterns;
		let num_leaves = self.num_leaves;

		for (transition, leaf) in transitions
			.iter()
			.zip(nodes.iter().copied())
			.take(leaves_end)
		{
			self.selectors[leaf] ^= 1;

			let range = self.slice(leaf);
			let projections = &mut self.projections[range];

			let offset = leaf * num_patterns;
			let samples =
				&self.samples[offset..offset + num_patterns];

			for (sample, projection) in
				samples.iter().copied().zip(projections)
			{
				*projection = calc_sample_projection(
					sample, transition,
				);
			}
		}

		for ((node, transition), (left_edge, right_edge)) in nodes
			.iter()
			.copied()
			.zip(transitions)
			.skip(leaves_end)
			.zip(children)
		{
			self.selectors[node] ^= 1;

			// SAFETY: TODO
			let node_projections = unsafe {
				from_raw_parts_mut(
					self.offset_ptr(node),
					num_patterns,
				)
			};

			// SAFETY: TODO
			let left_projections = unsafe {
				from_raw_parts(
					self.offset_ptr(*left_edge),
					num_patterns,
				)
			};
			// SAFETY: TODO
			let right_projections = unsafe {
				from_raw_parts(
					self.offset_ptr(*right_edge),
					num_patterns,
				)
			};

			// SAFETY: TODO
			let new_scales = unsafe {
				from_raw_parts_mut(
					self.scales
						.as_mut_ptr()
						.add(self.offset(node)),
					num_patterns,
				)
			};
			// SAFETY: TODO
			let old_scales = unsafe {
				from_raw_parts(
					self.scales
						.as_mut_ptr()
						.add(self.offset_old(node)),
					num_patterns,
				)
			};

			let scales_iter = new_scales.iter_mut().zip(old_scales);

			for (
				pattern,
				(((left, right), node), (new_scale, old_scale)),
			) in left_projections
				.iter()
				.zip(right_projections)
				.zip(node_projections)
				.zip(scales_iter)
				.enumerate()
			{
				let prod = hadamard(left, right);
				let mut projection = [0.0; N];

				for i in 0..N {
					projection[i] =
						dot(&prod, &transition[i]);
				}

				let should_scale =
					lt(&projection, self.scale_threshold);
				if should_scale {
					mul(&mut projection, self.scale_mult);
				}
				if should_scale != *old_scale {
					if should_scale {
						self.scale_sums[pattern] +=
							self.scale_ln;
					} else {
						self.scale_sums[pattern] -=
							self.scale_ln;
					}
				}
				*new_scale = should_scale;

				*node = projection;
			}
		}

		let root = *nodes.last().unwrap();
		let (root_left_edge, root_right_edge) =
			children.last().unwrap();

		let left_range = self.slice(*root_left_edge);
		let right_range = self.slice(*root_right_edge);

		let lefts = &self.projections[left_range];
		let rights = &self.projections[right_range];

		for (i, (left, right)) in lefts.iter().zip(rights).enumerate() {
			let likelihood = hadamard(left, right);
			let likelihood = hadamard(&likelihood, &frequencies);
			let sum: f64 = likelihood.iter().sum();
			let ln_sum = sum.ln();

			self.likelihoods[i] = ln_sum;
		}
		self.clear_scales(root);

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
		self.selectors_backup.copy_from_slice(&self.selectors);
		self.scale_sums_backup.copy_from_slice(&self.scale_sums);

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.selectors.copy_from_slice(&self.selectors_backup);
		self.scale_sums.copy_from_slice(&self.scale_sums_backup);

		Ok(())
	}
}

impl<const N: usize, F> CpuLikelihood<N, F> {
	fn offset(&self, idx: usize) -> usize {
		(idx * 2 + self.selectors[idx] as usize) * self.num_patterns
	}

	fn offset_old(&self, idx: usize) -> usize {
		let selector = (self.selectors[idx] ^ 1) as usize;
		(idx * 2 + selector) * self.num_patterns
	}

	/// # Safety
	///
	/// `idx` must be within `num_nodes`
	unsafe fn offset_ptr(&mut self, idx: usize) -> *mut [F; N] {
		// SAFETY: because of the invariant `offset(idx)` will be within
		// bounds
		unsafe { self.projections.as_mut_ptr().add(self.offset(idx)) }
	}

	/// # Safety
	///
	/// `idx` must be within `num_nodes`
	unsafe fn offset_old_ptr(&mut self, idx: usize) -> *mut [F; N] {
		// SAFETY: because of the invariant `offset(idx)` will be within
		// bounds
		unsafe { self.projections.as_mut_ptr().add(self.offset(idx)) }
	}

	fn slice(&self, idx: usize) -> Range<usize> {
		let offset = self.offset(idx);
		offset..offset + self.num_patterns
	}

	fn clear_scales(&mut self, node: usize) {
		let range = self.slice(node);
		let scales_old = &mut self.scales[range];
		for (pattern, scale) in scales_old.iter().enumerate() {
			if *scale {
				self.scale_sums[pattern] -= self.scale_ln;
			}
		}

		self.selectors[node] ^= 1;

		let range = self.slice(node);
		let scales_new = &mut self.scales[range];
		scales_new.fill(false);
	}
}

fn calc_sample_projection<const N: usize>(
	sample: u8,
	transition: &[[f64; N]; N],
) -> [f64; N] {
	let mut out = [0.0; N];

	if sample & 0b0001 != 0 {
		out[0] += transition[0][0];
		out[1] += transition[1][0];
		out[2] += transition[2][0];
		out[3] += transition[3][0];
	}
	if sample & 0b0010 != 0 {
		out[0] += transition[0][1];
		out[1] += transition[1][1];
		out[2] += transition[2][1];
		out[3] += transition[3][1];
	}
	if sample & 0b0100 != 0 {
		out[0] += transition[0][2];
		out[1] += transition[1][2];
		out[2] += transition[2][2];
		out[3] += transition[3][2];
	}
	if sample & 0b1000 != 0 {
		out[0] += transition[0][3];
		out[1] += transition[1][3];
		out[2] += transition[2][3];
		out[3] += transition[3][3];
	}

	out
}

fn hadamard<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
	let mut out = [0.0; N];

	for i in 0..N {
		out[i] = a[i] * b[i];
	}

	out
}

fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
	let mut out = a[0] * b[0];

	for i in 1..N {
		out += a[i] * b[i];
	}

	out
}

fn lt<const N: usize>(v: &[f64; N], threshold: f64) -> bool {
	v.iter().all(|&v| v < threshold)
}

fn mul<const N: usize>(v: &mut [f64; N], mult: f64) {
	for el in v {
		*el *= mult;
	}
}

impl CpuLikelihood<4, f64> {
	pub fn new(
		num_patterns: usize,
		leaves: Vec<u8>,
		scale_ln: u32,
	) -> Self {
		let num_leaves = leaves.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let projections =
			vec![Default::default(); num_nodes * num_patterns * 2];
		let scales = vec![false; num_nodes * num_patterns * 2];
		let scale_sums = vec![0; num_patterns];

		let scale_ln_f64: f64 = scale_ln.into();
		let scale_mult = scale_ln_f64.exp();
		let scale_threshold = (-scale_ln_f64).exp();

		Self {
			num_patterns,
			num_leaves,

			selectors: vec![0; num_nodes],
			selectors_backup: vec![0; num_nodes],

			samples: leaves,
			projections,

			likelihoods: vec![f64::NAN; num_patterns],

			scales,
			scale_sums_backup: scale_sums.clone(),
			scale_sums,

			scale_ln,
			scale_threshold,
			scale_mult,
		}
	}
}
