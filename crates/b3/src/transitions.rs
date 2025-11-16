use anyhow::Result;
use pyo3::prelude::*;

use crate::{
	clock::PyClock, likelihood::Space,
	substitution::BoxedSubstitutionModel, tree::Tree,
};
use skvec::SkVec;

pub struct Transitions<S: Space> {
	substitution: BoxedSubstitutionModel<S>,
	clock: PyClock,

	rate: f64,

	transitions: SkVec<S::Matrix>,
}

impl<S: Space> Transitions<S> {
	pub fn new(
		length: usize,
		substitution: BoxedSubstitutionModel<S>,
		clock: PyClock,
	) -> Self {
		let transitions = SkVec::repeat(S::Matrix::default(), length);

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
		let distances: Vec<S::Scalar> = edges
			.iter()
			.copied()
			.map(|e| -> S::Scalar {
				let distance_f64 =
					tree.edge_length(e) * self.rate;
				distance_f64.into()
			})
			.collect();

		self.update_edges(&edges, &distances);

		Ok(full_update)
	}

	fn update_edges(&mut self, edges: &[usize], distances: &[S::Scalar]) {
		for (edge, distance) in edges.iter().zip(distances) {
			let transition =
				self.substitution.get_transition(*distance);

			self.transitions.set(*edge, transition);
		}
	}

	pub fn accept(&mut self) {
		self.transitions.accept();
	}

	pub fn reject(&mut self) {
		self.transitions.reject();
	}

	pub fn matrices(&self, edges: &[usize]) -> Vec<S::Matrix> {
		let mut out = Vec::with_capacity(edges.len());

		for edge in edges {
			out.push(self.transitions[*edge])
		}

		out
	}

	pub fn frequencies(&self) -> S::Vector {
		self.substitution.get_frequencies()
	}
}
