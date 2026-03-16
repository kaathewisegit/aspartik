use anyhow::Result;
use num_traits::Zero;
use pyo3::prelude::*;

use crate::{
	clock::PyClock, parameters::Tree, substitution::SubstitutionModel,
};
use skvec::{SkVec, skvec};

pub struct Transitions<const N: usize, F> {
	substitution: Box<dyn SubstitutionModel<N, F> + Sync + Send>,
	clock: Py<PyClock>,

	transitions: SkVec<[[F; N]; N]>,
}

impl<const N: usize, F> Transitions<N, F>
where
	F: Default + Copy + Zero + Default,
{
	pub fn new<S>(
		length: usize,
		substitution: S,
		clock: Py<PyClock>,
	) -> Self
	where
		S: SubstitutionModel<N, F> + Sync + Send + 'static,
	{
		let transitions = skvec![[[F::zero(); N]; N]; length];

		Self {
			substitution: Box::new(substitution),
			clock,

			transitions,
		}
	}

	/// Returns `true` if a full update is needed.
	pub fn update(&mut self, tree: &mut Tree) -> Result<()> {
		let mut clock = self.clock.get().inner();
		clock.update()?;
		clock.mark_tree(tree);

		if self.substitution.update()? {
			tree.mark_all_edges_updated();
		}

		for edge in tree.edges_to_update() {
			let rate = clock.get_rate(edge);
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
		self.substitution.accept();
		self.clock.get().inner().accept();
	}

	pub fn reject(&mut self) {
		self.transitions.reject();
		self.substitution.reject();
		self.clock.get().inner().reject();
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
