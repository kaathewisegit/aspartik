use crate::substitution::Substitution;
use linalg::RowMatrix;
use skvec::SkVec;

use crate::tree::Tree;

pub struct Transitions<const N: usize> {
	current: Box<Substitution<N>>,

	p: Box<RowMatrix<f64, N, N>>,
	diag: Box<RowMatrix<f64, N, N>>,
	inv_p: Box<RowMatrix<f64, N, N>>,

	rate: f64,

	transitions: SkVec<RowMatrix<f64, N, N>>,
}

impl<const N: usize> Transitions<N> {
	pub fn new(length: usize) -> Self {
		let transitions = SkVec::repeat(RowMatrix::default(), length);

		Self {
			current: Box::new(RowMatrix::default()),

			p: Box::new(RowMatrix::default()),
			diag: Box::new(RowMatrix::default()),
			inv_p: Box::new(RowMatrix::default()),

			rate: 1.0,

			transitions,
		}
	}

	/// Returns `true` if a full update is needed.
	pub fn update(
		&mut self,
		substitution: Substitution<N>,
		rate: f64,
		tree: &Tree,
	) -> bool {
		let full_update =
			substitution != *self.current || rate != self.rate;
		if full_update {
			self.current = Box::new(substitution);
			self.rate = rate;

			let (eigenvectors, eigenvalues) = substitution.eigen();

			self.diag =
				Box::new(RowMatrix::from_diagonal(eigenvalues));
			self.p = Box::new(eigenvectors);
			self.inv_p = Box::new(self.p.inverse());
		}

		let edges: Vec<usize> = if full_update {
			(0..(tree.num_internals() * 2)).collect()
		} else {
			tree.edges_to_update()
		};
		let distances: Vec<f64> = edges
			.iter()
			.copied()
			.map(|e| tree.edge_distance(e) * rate)
			.collect();

		self.update_edges(&edges, &distances);

		full_update
	}

	fn update_edges(&mut self, edges: &[usize], distances: &[f64]) {
		let inv_p = *self.inv_p;
		let p = *self.p;
		for (edge, distance) in edges.iter().zip(distances) {
			let diag = self
				.diag
				.map_diagonal(|v| (v * distance).exp());

			let transition = p * diag * inv_p;

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
