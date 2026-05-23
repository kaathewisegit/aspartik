#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim {
	pub rows: u32,
	pub cols: u32,
	pub row_stride: u32,
}

impl Dim {
	pub(crate) const fn is_index_valid(
		&self,
		row: usize,
		col: usize,
	) -> bool {
		row < (self.rows as usize) && col < (self.cols as usize)
	}

	pub(crate) const fn offset(&self, row: usize, col: usize) -> usize {
		row * (self.row_stride as usize) + col
	}

	pub const fn is_square(&self) -> bool {
		self.rows == self.cols
	}

	pub const fn num_slots(&self) -> usize {
		(self.rows * self.row_stride) as usize
	}
}
