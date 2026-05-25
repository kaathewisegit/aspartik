use crate::{MatrixMut, MatrixRef};

pub trait MatrixArrayExt {
	type T;

	fn as_mat(&self) -> MatrixRef<'_, Self::T>;
	fn as_mat_mut(&mut self) -> MatrixMut<'_, Self::T>;
}

impl<T, const N: usize, const M: usize> MatrixArrayExt for [[T; M]; N] {
	type T = T;

	fn as_mat(&self) -> MatrixRef<'_, T> {
		MatrixRef::from_array(self)
	}

	fn as_mat_mut(&mut self) -> MatrixMut<'_, T> {
		MatrixMut::from_array(self)
	}
}

pub trait MatrixSliceExt {
	type T;

	fn as_mat(&self, cols: usize, rows: usize) -> MatrixRef<'_, Self::T>;
	fn as_mat_mut(
		&mut self,
		cols: usize,
		rows: usize,
	) -> MatrixMut<'_, Self::T>;
}

impl<T> MatrixSliceExt for [T] {
	type T = T;

	fn as_mat(&self, cols: usize, rows: usize) -> MatrixRef<'_, T> {
		MatrixRef::from_slice(self, rows, cols)
	}

	fn as_mat_mut(&mut self, cols: usize, rows: usize) -> MatrixMut<'_, T> {
		MatrixMut::from_slice(self, rows, cols)
	}
}
