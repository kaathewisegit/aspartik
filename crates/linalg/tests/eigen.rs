use approx::assert_relative_eq;
use proptest::prelude::*;

use linalg::{proptest::symmetric, RowMatrix};

#[test]
fn roundtrip() {
	let jc = RowMatrix::from([
		[-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
		[1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0 / 3.0],
		[1.0 / 3.0, 1.0 / 3.0, -1.0, 1.0 / 3.0],
		[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0],
	]);

	let diag = RowMatrix::from_diagonal(jc.eigenvalues());
	let eigenvectors = jc.eigenvectors();
	let inverse = eigenvectors.inverse();

	assert_relative_eq!(jc, inverse * diag * eigenvectors);
	assert_relative_eq!(jc * 0.1, inverse * (diag * 0.1) * eigenvectors);
}

proptest! {
	#[test]
	fn symmetric_eigen_2(m in symmetric::<2>()) {
		let eigenvalues = m.eigenvalues();
		let eigenvectors = m.eigenvectors();

		for i in 0..2 {
			assert_relative_eq!(
				m * eigenvectors[i],
				eigenvectors[i] * eigenvalues[i],
				max_relative = 1e-10,
			);
		}
	}

	// TODO: hermetic matrices

	#[test]
	fn inverse_2(m in symmetric::<2>()) {
		let diag = RowMatrix::from_diagonal(m.eigenvalues());
		let eigenvectors = m.eigenvectors();
		let inverse = eigenvectors.inverse();

		assert_relative_eq!(
			m,
			inverse * diag * eigenvectors,
			max_relative = 1e-10,
		);
	}
}
