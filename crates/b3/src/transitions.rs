use anyhow::Result;
use pyo3::prelude::*;

use crate::{substitution::SubstitutionModel, tree::Tree};
use linalg::RowMatrix;
use skvec::SkVec;

pub struct Transitions<const N: usize> {
	rate: f64,

	transitions: SkVec<RowMatrix<f64, N, N>>,
}

impl<const N: usize> Transitions<N> {
	pub fn new(length: usize) -> Self {
		let transitions = SkVec::repeat(RowMatrix::default(), length);

		Self {
			rate: 1.0,

			transitions,
		}
	}

	/// Returns `true` if a full update is needed.
	pub fn update(
		&mut self,
		py: Python,
		substitution: &mut dyn SubstitutionModel<N>,
		rate: f64,
		tree: &Tree,
	) -> Result<bool> {
		let full_update = substitution.update(py)? || rate != self.rate;

		let edges: Vec<usize> = if full_update {
			(0..(tree.num_internals() * 2)).collect()
		} else {
			tree.edges_to_update()
		};
		let distances: Vec<f64> = edges
			.iter()
			.copied()
			.map(|e| tree.edge_length(e) * rate)
			.collect();

		self.update_edges(&edges, &distances, substitution);

		Ok(full_update)
	}

	fn update_edges(
		&mut self,
		edges: &[usize],
		distances: &[f64],
		substitution: &dyn SubstitutionModel<N>,
	) {
		for (edge, distance) in edges.iter().zip(distances) {
			let transition = substitution.get_transition(*distance);

			self.transitions.set(*edge, transition);
		}
	}

	pub fn accept(&mut self) {
		self.transitions.accept();
	}

	pub fn reject(&mut self) {
		self.transitions.reject();
	}

	pub fn matrices(&self, edges: &[usize]) -> Vec<RowMatrix<f64, N, N>> {
		let mut out = Vec::with_capacity(edges.len());

		for edge in edges {
			out.push(self.transitions[*edge])
		}

		out
	}
}
