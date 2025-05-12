use crate::substitution::Substitution;
use linalg::RowMatrix;
use skvec::SkVec;

use crate::tree::Tree;

pub struct Transitions<const N: usize> {
	current: Substitution<N>,

	p: RowMatrix<f64, N, N>,
	diag: RowMatrix<f64, N, N>,
	inv_p: RowMatrix<f64, N, N>,

	transitions: SkVec<RowMatrix<f64, N, N>>,
}

impl<const N: usize> Transitions<N> {
	pub fn new(length: usize) -> Self {
		let transitions = SkVec::repeat(RowMatrix::default(), length);

		Self {
			current: Default::default(),

			p: Default::default(),
			diag: Default::default(),
			inv_p: Default::default(),

			transitions,
		}
	}

	/// Returns `true` if a full update is needed.
	pub fn update(
		&mut self,
		substitution: Substitution<N>,
		tree: &Tree,
	) -> bool {
		let full_update = substitution != self.current;
		if full_update {
			self.current = substitution;

			self.diag = RowMatrix::from_diagonal(
				substitution.eigenvalues(),
			);
			self.p = substitution.eigenvectors();
			self.inv_p = self.p.inverse();
		}

		let edges: Vec<usize> = if full_update {
			(0..(tree.num_internals() * 2)).collect()
		} else {
			tree.edges_to_update()
		};
		let distances: Vec<f64> = edges
			.iter()
			.copied()
			.map(|e| tree.edge_distance(e))
			.collect();

		self.update_edges(&edges, &distances);

		full_update
	}

	fn update_edges(&mut self, edges: &[usize], distances: &[f64]) {
		for (edge, distance) in edges.iter().zip(distances) {
			let diag = self
				.diag
				.map_diagonal(|v| (v * distance).exp());

			let transition = self.inv_p * diag * self.p;

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
