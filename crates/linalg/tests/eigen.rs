use arbtest::arbtest;
use math::assert_almost_eq;

use linalg::{RowMatrix, arbitrary::symmetric};

#[test]
fn roundtrip() {
	let jc = RowMatrix::from([
		[-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
		[1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0 / 3.0],
		[1.0 / 3.0, 1.0 / 3.0, -1.0, 1.0 / 3.0],
		[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0],
	]);

	let (eigenvalues, eigenvectors) = jc.eigen();
	let diag = RowMatrix::from_diagonal(eigenvectors);
	let inverse = eigenvalues.inverse();

	assert_almost_eq!(jc, eigenvalues * diag * inverse);
	assert_almost_eq!(jc * 0.1, eigenvalues * (diag * 0.1) * inverse);
}

#[ignore]
#[test]
fn symmetric_eigen_2() {
	arbtest(|u| {
		let m: RowMatrix<f64, 2, 2> = symmetric(u)?;
		let (eigenvalues, eigenvectors) = m.eigen();

		for i in 0..2 {
			let lambda = eigenvectors[i];
			let v = eigenvalues[i];
			assert_almost_eq!(m * v, v * lambda, relative = 1e-13,);
		}

		Ok(())
	});
}

// TODO: hermetic matrices

#[test]
fn inverse_2() {
	arbtest(|u| {
		let m: RowMatrix<f64, 2, 2> = symmetric(u)?;
		let (eigenvalues, eigenvectors) = m.eigen();
		let diag = RowMatrix::from_diagonal(eigenvectors);
		let inverse = eigenvalues.inverse();

		assert_almost_eq!(
			m,
			eigenvalues * diag * inverse,
			// 0x98d8990800010000 for 1e-9
			relative = 1e-8,
		);

		Ok(())
	});
}
