use num_traits::Float;

use crate::{RowMatrix, Vector};

impl<F: Float, const N: usize> Vector<F, N> {
	pub fn abs(&self) -> Self {
		self.map(|e| e.abs())
	}
}

impl<F: Float, const N: usize, const M: usize> RowMatrix<F, N, M> {
	pub fn abs(&self) -> Self {
		self.map(|e| e.abs())
	}
}
