use anyhow::Result;
use linalg::RowMatrix;
use pyo3::prelude::*;

use crate::{clock::PyClock, substitution::BoxedSubstitutionModel, tree::Tree};
use skvec::SkVec;

pub struct Transitions<const N: usize, F> {
	substitution: BoxedSubstitutionModel<N, F>,
	clock: PyClock,

	rate: f64,

	transitions: SkVec<RowMatrix<F, N, N>>,
}

impl<const N: usize, F> Transitions<N, F>
where
	F: Default + Copy,
{
	pub fn new(
		length: usize,
		substitution: BoxedSubstitutionModel<N, F>,
		clock: PyClock,
	) -> Self {
		let transitions = SkVec::repeat(RowMatrix::default(), length);

		Self {
			substitution,
			clock,

			rate: f64::NAN,

			transitions,
		}
	}

	/// Returns `true` if a full update is needed.
	pub fn update(&mut self, py: Python, tree: &Tree) -> Result<bool> {
		let new_rate = self.clock.get_rate(py)?;

		let full_update =
			self.substitution.update(py)? || self.rate != new_rate;
		if full_update {
			self.rate = new_rate;
		}

		let edges: Vec<usize> = if full_update {
			(0..(tree.num_internals() * 2)).collect()
		} else {
			tree.edges_to_update()
		};
		let distances: Vec<f64> = edges
			.iter()
			.copied()
			.map(|e| tree.edge_length(e) * self.rate)
			.collect();

		self.update_edges(&edges, &distances);

		Ok(full_update)
	}

	fn update_edges(&mut self, edges: &[usize], distances: &[f64]) {
		for (edge, distance) in edges.iter().zip(distances) {
			let transition =
				self.substitution.get_transition(*distance);

			self.transitions.set(*edge, transition.into());
		}
	}

	pub fn accept(&mut self) {
		self.transitions.accept();
	}

	pub fn reject(&mut self) {
		self.transitions.reject();
	}

	pub fn matrices(&self, edges: &[usize]) -> Vec<[[F; N]; N]> {
		let mut out = Vec::with_capacity(edges.len());

		for edge in edges {
			out.push(self.transitions[*edge].into())
		}

		out
	}

	pub fn frequencies(&self) -> [F; N] {
		self.substitution.get_frequencies()
	}
}
