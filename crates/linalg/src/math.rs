use nalgebra::{Const, DMatrix, SymmetricEigen};

use crate::{RowMatrix, Vector};

type Nalg<const N: usize> = nalgebra::OMatrix<f64, Const<N>, Const<N>>;

impl<const N: usize> RowMatrix<f64, N, N> {
	pub fn decompose(
		&self,
	) -> (Vector<f64, N>, RowMatrix<f64, N, N>, RowMatrix<f64, N, N>) {
		if !self.is_symmetric() {
			unimplemented!("Non-symmetric eigendecomposition");
		}

		let m: Nalg<N> = self.into();

		// an ugly workaround because `SymmetricEigen` can't accept
		// arbitrary Const<N>, only concrete ones.
		let vec: Vec<f64> = m.into_iter().copied().collect();
		let m = DMatrix::from_vec(N, N, vec);

		let s = SymmetricEigen::new(m);

		let slice = s.eigenvalues.as_slice();
		let array: [f64; N] = slice.try_into().unwrap();

		(
			array.into(),
			s.eigenvectors.clone().into(),
			s.eigenvectors.try_inverse().unwrap().into(),
		)
	}

	pub fn eigenvalues(&self) -> Vector<f64, N> {
		self.decompose().0
	}

	pub fn eigenvectors(&self) -> RowMatrix<f64, N, N> {
		self.decompose().1
	}

	pub fn inverse(&self) -> Self {
		let m: Nalg<N> = self.into();
		m.try_inverse().unwrap().into()
	}
}
