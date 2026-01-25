use anyhow::Result;
use num_traits::Zero;
use pyo3::prelude::*;

use crate::{clock::PyClock, substitution::BoxedSubstitutionModel, tree::Tree};
use skvec::{SkVec, skvec};

pub struct Transitions<const N: usize, F> {
	substitution: BoxedSubstitutionModel<N, F>,
	clock: PyClock,

	rate: f64,

	transitions: SkVec<[[F; N]; N]>,
}

impl<const N: usize, F> Transitions<N, F>
where
	F: Default + Copy + Zero + Default,
{
	pub fn new(
		length: usize,
		substitution: BoxedSubstitutionModel<N, F>,
		clock: PyClock,
	) -> Self {
		let transitions = skvec![[[F::zero(); N]; N]; length];

		Self {
			substitution,
			clock,

			rate: f64::NAN,

			transitions,
		}
	}

	/// Returns `true` if a full update is needed.
	pub fn update(&mut self, py: Python, tree: &mut Tree) -> Result<()> {
		let new_rate = self.clock.get_rate(py)?;
		if self.rate != new_rate {
			tree.mark_all_edges_updated();
			self.rate = new_rate;
		}

		if self.substitution.update(py)? {
			tree.mark_all_edges_updated();
		}

		for edge in tree.edges_to_update() {
			let rate = self.rate;
			let time_length = tree.edge_length(edge);
			let length = time_length * rate;

			let transition =
				self.substitution.get_transition(length);

			self.transitions.set(edge, transition);
		}

		Ok(())
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
			out.push(self.transitions[*edge])
		}

		out
	}

	pub fn frequencies(&self) -> [F; N] {
		self.substitution.get_frequencies()
	}
}
