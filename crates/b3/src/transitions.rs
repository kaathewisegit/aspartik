use anyhow::Result;
use num_traits::Zero;

use crate::{clock::Clock, parameters::Tree, substitution::SubstitutionModel};
use sk::{SkBuf, skbuf};

pub struct Transitions<const N: usize, F> {
	transitions: SkBuf<[[F; N]; N]>,
}

impl<const N: usize, F> Transitions<N, F>
where
	F: Default + Copy + Zero + Default,
{
	pub fn new(length: usize) -> Self {
		let transitions = skbuf![[[F::zero(); N]; N]; length];

		Self { transitions }
	}

	/// Returns `true` if a full update is needed.
	pub fn update(
		&mut self,
		tree: &mut Tree,
		clock: &Clock,
		substitution: &dyn SubstitutionModel<N, F>,
	) -> Result<()> {
		for edge in tree.edges_to_update() {
			let rate = clock.get_rate(edge);
			let time_length = tree.edge_length(edge);
			let length = time_length * rate;

			let transition = substitution.get_transition(length);

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
}
