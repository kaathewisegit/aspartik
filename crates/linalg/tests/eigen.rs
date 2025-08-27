use approx::assert_relative_eq;
use arbtest::arbtest;

use linalg::{RowMatrix, arbitrary::symmetric};

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

#[test]
fn symmetric_eigen_2() {
	arbtest(|u| {
		let m: RowMatrix<f64, 2, 2> = symmetric(u)?;
		let eigenvalues = m.eigenvalues();
		let eigenvectors = m.eigenvectors();

		for i in 0..2 {
			assert_relative_eq!(
				m * eigenvectors[i],
				eigenvectors[i] * eigenvalues[i],
				// 0xe4b8998f00010000 for 1e-15
				// 0x3aaa44be00000236 for 1e-14
				max_relative = 1e-13,
			);
		}

		Ok(())
	});
}

// TODO: hermetic matrices

#[test]
fn inverse_2() {
	arbtest(|u| {
		let m: RowMatrix<f64, 2, 2> = symmetric(u)?;
		let diag = RowMatrix::from_diagonal(m.eigenvalues());
		let eigenvectors = m.eigenvectors();
		let inverse = eigenvectors.inverse();

		assert_relative_eq!(
			m,
			inverse * diag * eigenvectors,
			// 0xaac90ff800010000 for 1e-14
			max_relative = 1e-13,
		);

		Ok(())
	});
}
