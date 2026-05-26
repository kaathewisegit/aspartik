use bytemuck::Zeroable;

use std::ops::Index;

use buffer::SliceBuffer;
use sk::EditBuf;

pub struct SkSliceBuf<T> {
	edits: EditBuf,
	slices: SliceBuffer<T>,
}

impl<T> SkSliceBuf<T> {
	pub fn new(size: usize, len: usize) -> Self
	where
		T: Zeroable,
	{
		Self {
			edits: EditBuf::new(len),
			slices: SliceBuffer::new(size, len * 2),
		}
	}

	/// Marks `index`th element as edited and returns a slice to it
	pub fn update(&mut self, index: usize) -> &mut [T] {
		self.edits.set_edited(index);
		let bit = self.edits.offset(index);

		&mut self.slices[index * 2 + bit]
	}

	pub fn accept(&mut self) {
		self.edits.accept();
	}

	pub fn reject(&mut self) {
		self.edits.reject();
	}
}

impl<T> Index<usize> for SkSliceBuf<T> {
	type Output = [T];

	fn index(&self, index: usize) -> &[T] {
		&self.slices[index * 2 + self.edits.offset(index)]
	}
}
