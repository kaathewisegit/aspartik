use linalg::{
	lapack::{eigen, inverse},
	matrix::{Matrix, SquareMatrix},
};
use math::assert_almost_eq;

#[test]
fn reconstruction() {
	type M = [[f64; 2]; 2];

	let m = [[1.0, 2.0], [3.0, 4.0]];

	let (values, vectors) = eigen(&m);
	let inv_vectors = inverse(&vectors);

	let diag: M = SquareMatrix::from_diagonal(values);

	let reconstructed: M = vectors.mul::<_, M>(diag).mul(inv_vectors);

	for i in 0..2 {
		for j in 0..2 {
			assert_almost_eq!(*reconstructed.at(i, j), *m.at(i, j));
		}
	}
}
