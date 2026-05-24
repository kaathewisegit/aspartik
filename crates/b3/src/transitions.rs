use anyhow::Result;
use linalg::MatrixMut;

use std::ops::Range;

use crate::{parameters::Tree, substitution::SubstitutionModel};
use sk::EditBuf;

pub struct Transitions {
	size: usize,
	edits: EditBuf,
	data: Box<[f64]>,
	frequencies: Box<[f64]>,
}

impl Transitions {
	pub fn new(size: usize, len: usize) -> Self {
		let data = vec![0.0; size * size * len * 2].into_boxed_slice();
		let frequencies = vec![0.0; size].into_boxed_slice();
		let edits = EditBuf::new(len);

		Self {
			size,
			edits,
			data,
			frequencies,
		}
	}

	fn get_range(&self, edge: usize) -> Range<usize> {
		let ms = self.size * self.size;
		let bit = self.edits.offset(edge);

		let offset = ms * (edge * 2 + bit);
		offset..offset + ms
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

			self.edits.set_edited(edge);

			let range = self.get_range(edge);
			let m_ref = MatrixMut::from_slice(
				&mut self.data[range],
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
		self.edits.accept();
	}

	pub fn reject(&mut self) {
		self.edits.reject();
	}

	pub fn write_matrices(&self, edges: &[usize], dst: &mut [f64]) {
		let size = self.size;
		let ms = size * size;

		let mut offset = 0;
		for edge in edges {
			let range = self.get_range(*edge);
			let src_slice = &self.data[range];

			let dst_slice = &mut dst[offset..offset + ms];
			dst_slice.copy_from_slice(src_slice);
			offset += ms;
		}
	}
}
