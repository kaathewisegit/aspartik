use anyhow::Result;
use linalg::MatrixMut;

use crate::{SkSliceBuf, parameters::Tree, substitution::SubstitutionModel};

pub struct Transitions {
	size: usize,
	data: SkSliceBuf<f64>,
	frequencies: Box<[f64]>,
}

impl Transitions {
	pub fn new(size: usize, len: usize) -> Self {
		let data = SkSliceBuf::new(size * size, len);
		let frequencies = vec![0.0; size].into_boxed_slice();

		Self {
			size,
			data,
			frequencies,
		}
	}

	pub fn update(
		&mut self,
		tree: &mut Tree,
		substitution: &mut dyn SubstitutionModel,
		clock_rate: impl Fn(usize) -> f64,
	) -> Result<()> {
		let size = self.size;
		for edge in tree.edges_to_update() {
			let time_length = tree.edge_length(edge);
			let length = time_length * clock_rate(edge);

			let m_ref = MatrixMut::from_slice(
				self.data.update(edge),
				size,
				size,
			);
			substitution.write_transition(length, m_ref);
		}

		substitution.write_frequencies(&mut self.frequencies);

		Ok(())
	}

	pub fn frequencies(&self) -> &[f64] {
		&self.frequencies
	}

	pub fn accept(&mut self) {
		self.data.accept();
	}

	pub fn reject(&mut self) {
		self.data.reject();
	}

	pub fn write_matrices(&self, edges: &[usize], dst: &mut [f64]) {
		let size = self.size;
		let ms = size * size;

		let mut offset = 0;
		for &edge in edges {
			let dst_slice = &mut dst[offset..offset + ms];
			dst_slice.copy_from_slice(&self.data[edge]);
			offset += ms;
		}
	}
}
