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

	let (eigenvalues, r_eigenvectors, l_eigenvectors) = jc.decompose();
	let diag = RowMatrix::from_diagonal(eigenvalues);

	assert_almost_eq!(r_eigenvectors * diag * l_eigenvectors, jc, ulps = 7);
	assert_almost_eq!(
		r_eigenvectors * (diag * 0.1) * l_eigenvectors,
		jc * 0.1
	);
}

#[ignore]
#[test]
fn symmetric_eigen_2() {
	arbtest(|u| {
		let m: RowMatrix<f64, 2, 2> = symmetric(u)?;
		let (eigenvalues, eigenvectors, _) = m.decompose();

		for i in 0..2 {
			let lambda = eigenvalues[i];
			let v = eigenvectors[i];
			assert_almost_eq!(m * v, v * lambda, relative = 1e-13);
		}

		Ok(())
	});
}

// TODO: hermetic matrices

#[test]
fn inverse_2() {
	arbtest(|u| {
		let m: RowMatrix<f64, 2, 2> = symmetric(u)?;
		let (eigenvalues, r_eigenvectors, l_eigenvectors) =
			m.decompose();
		let diag = RowMatrix::from_diagonal(eigenvalues);

		assert_almost_eq!(
			m,
			r_eigenvectors * diag * l_eigenvectors,
			// 0x98d8990800010000 for 1e-9
			relative = 1e-8,
		);

		Ok(())
	});
}
