use arbtest::arbtest;

use linalg::{
	arbitrary::symmetric,
	lapack::{eigen, inverse},
	matrix::{Matrix, SquareMatrix},
};
use math::assert_almost_eq;

type M<const N: usize> = [[f64; N]; N];

fn reconstruct<const N: usize>(m: M<N>) {
	let (values, vectors) = eigen(&m);
	let inv_vectors: M<N> = inverse(&vectors);

	let diag: M<N> = SquareMatrix::from_diagonal(values);

	let reconstructed: M<N> =
		vectors.mul::<_, M<N>>(&diag).mul(&inv_vectors);

	for i in 0..N {
		for j in 0..N {
			assert_almost_eq!(
				*reconstructed.at(i, j),
				*m.at(i, j),
				relative = 1e-9
			);
		}
	}
}

#[test]
fn reconstruction_basic() {
	let m = [[1.0, 2.0], [3.0, 4.0]];
	reconstruct(m);
}

#[test]
fn reconstruction_symmetric() {
	arbtest(|u| {
		let m = symmetric::<4>(u)?;

		reconstruct(m);

		Ok(())
	})
	.size_min(128)
	.budget_ms(500);
}
