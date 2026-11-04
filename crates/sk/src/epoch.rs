use std::ops::{Deref, Index, IndexMut, Range};

#[derive(Debug, Clone)]
pub struct EpochBuf<T> {
	len: usize,
	values: Box<[T]>,
	indices: Vec<usize>,
}

impl<T: Copy> EpochBuf<T> {
	pub fn repeat(value: T, len: usize) -> Self {
		Self {
			len,
			values: vec![value; len * 2].into_boxed_slice(),
			indices: Vec::new(),
		}
	}

	pub fn accept(&mut self) {
		let len = self.len;
		for index in self.indices.iter().copied() {
			self.values[index + len] = self.values[index];
		}
		self.indices.clear();
	}

	pub fn reject(&mut self) {
		let len = self.len;
		for index in self.indices.iter().copied() {
			self.values[index] = self.values[index + len];
		}
		self.indices.clear();
	}

	pub fn optimize_indices(&mut self) {
		self.indices.sort_unstable();
		self.indices.dedup();
	}

	pub fn is_changed(&self) -> bool {
		!self.indices.is_empty()
	}

	pub fn changed_indices(&self) -> &[usize] {
		&self.indices
	}

	pub fn slice_mut(&mut self) -> &mut [T] {
		self.indices.extend(0..self.len);
		&mut self.values[..self.len]
	}
}

impl<T> Deref for EpochBuf<T> {
	type Target = [T];

	fn deref(&self) -> &[T] {
		&self.values[..self.len]
	}
}

impl<T> Index<usize> for EpochBuf<T> {
	type Output = T;

	fn index(&self, index: usize) -> &T {
		&self.values[index]
	}
}

impl<T> IndexMut<usize> for EpochBuf<T> {
	fn index_mut(&mut self, index: usize) -> &mut T {
		self.indices.push(index);
		&mut self.values[index]
	}
}

impl<T> Index<Range<usize>> for EpochBuf<T> {
	type Output = [T];

	fn index(&self, index: Range<usize>) -> &[T] {
		&self.values[index]
	}
}

impl<T> IndexMut<Range<usize>> for EpochBuf<T> {
	fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
		self.indices.extend(index.clone());
		&mut self.values[index]
	}
}

impl<T: Default + Copy> From<Vec<T>> for EpochBuf<T> {
	fn from(mut values: Vec<T>) -> Self {
		let len = values.len();
		values.reserve_exact(len * 2);
		values.resize(len * 2, T::default());
		values.copy_within(..len, len);
		Self {
			len,
			values: values.into_boxed_slice(),
			indices: Vec::new(),
		}
	}
}
