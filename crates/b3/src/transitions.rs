use anyhow::Result;
use linalg::MatrixMut;
use num_traits::Zero;

use crate::{
	parameters::Tree,
	substitution::{Substitution, SubstitutionModel},
};
use sk::{EditBuf, SkBuf, skbuf};

pub struct Transitions<const N: usize, F> {
	transitions: SkBuf<[[F; N]; N]>,
}

impl<const N: usize, F> Transitions<N, F>
where
	F: Copy + Zero,
{
	pub fn new(length: usize) -> Self {
		let transitions = skbuf![[[F::zero(); N]; N]; length];

		Self { transitions }
	}

	pub fn update(
		&mut self,
		tree: &mut Tree,
		substitution: &dyn SubstitutionModel<N, F>,
		clock_rate: impl Fn(usize) -> f64,
	) -> Result<()> {
		for edge in tree.edges_to_update() {
			let time_length = tree.edge_length(edge);
			let length = time_length * clock_rate(edge);

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

pub struct TransitionsDyn {
	size: usize,
	edits: EditBuf,
	data: Box<[f64]>,
}

impl TransitionsDyn {
	pub fn new(size: usize, len: usize) -> Self {
		let values = vec![0.0; size * size * len].into_boxed_slice();
		let edits = EditBuf::new(len);

		Self {
			size,
			data: values,
			edits,
		}
	}

	pub fn update(
		&mut self,
		tree: &mut Tree,
		substitution: &Substitution,
		clock_rate: impl Fn(usize) -> f64,
	) -> Result<()> {
		let size = self.size;
		let ms = size * size;
		for edge in tree.edges_to_update() {
			let time_length = tree.edge_length(edge);
			let length = time_length * clock_rate(edge);

			self.edits.set_edited(edge);
			let bit = self.edits.offset(edge);

			let offset = ms * (edge * 2 + bit);
			let slice = &mut self.data[offset..offset + ms];
			let m_ref = MatrixMut::from_slice(slice, size, size);

			substitution.write_transition(length, m_ref);
		}

		Ok(())
	}

	pub fn accept(&mut self) {
		self.edits.accept();
	}

	pub fn reject(&mut self) {
		self.edits.reject();
	}
}
