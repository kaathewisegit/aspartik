use anyhow::Result;
use bytemuck::{allocation::cast_vec, cast_slice};
use num_traits::{Float, Inv, Num, NumAssign};

use std::{ops::Mul, slice};

use crate::transitions;

use super::LikelihoodTrait;
use linalg::{RowMatrix, Vector};
use skvec::{SkVec, skvec};

pub struct CpuLikelihood<const N: usize, F> {
	leaves: Vec<u8>,
	projections: SkVec<Vector<F, N>>,

	num_sites: usize,
	num_leaves: usize,

	updated_edges: Vec<usize>,
	likelihoods: Vec<f64>,

	scales: SkVec<bool>,
	scale_sums: SkVec<u32>,

	scale: F,
	inv_scale: F,
	scale_ln: u32,
}

impl<const N: usize, F> LikelihoodTrait<N, F> for CpuLikelihood<N, F>
where
	F: Float + Num + NumAssign,
	f64: From<F> + From<u32>,
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

		let transitions = cast_transitions(transitions);

		self.updated_edges = edges.to_vec();

		let num_sites = self.num_sites;
		let num_leaves = self.num_leaves;

		for i in 0..leaves_end {
			let transition = &transitions[i];

			let edge = edges[i];
			let edge_idx = edge * num_sites;

			let leaf = nodes[i];
			let leaf_idx = leaf * num_sites;

			for site in 0..num_sites {
				let leaf = self.leaves[leaf_idx + site];
				let projection =
					calc_leaf_projection(transition, leaf);

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

	fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()> {
		patterns.copy_from_slice(&self.likelihoods);

		for (i, scale_sum) in self.scale_sums.iter().enumerate() {
			patterns[i] -= f64::from(*scale_sum);
		}

		Ok(())
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

pub fn calc_leaf_projection<const N: usize, F: Num + NumAssign + Copy>(
	transition: &RowMatrix<F, N, N>,
	leaf: u8,
) -> Vector<F, N> {
	let mut out = Vector::zeros();

	if leaf & 0b0001 != 0 {
		out += transition[0];
	}
	if leaf & 0b0010 != 0 {
		out += transition[1];
	}
	if leaf & 0b0100 != 0 {
		out += transition[2];
	}
	if leaf & 0b1000 != 0 {
		out += transition[3];
	}

	out
}

pub fn cast_transitions<const N: usize, F>(
	transitions: &[[[F; N]; N]],
) -> &[RowMatrix<F, N, N>] {
	unsafe {
		slice::from_raw_parts(
			transitions.as_ptr() as *const _,
			transitions.len(),
		)
	}
}

impl CpuLikelihood<4, f64> {
	pub fn new(num_sites: usize, leaves: Vec<u8>, scale_ln: u32) -> Self {
		let num_leaves = leaves.len() / num_sites;
		let num_internals = num_leaves - 1;
		let num_edges = num_internals * 2;

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
