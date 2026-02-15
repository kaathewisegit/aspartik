use linalg::RowMatrix;
use math::assert_almost_eq;

#[test]
fn reconstruction() {
	let m = linalg::RowMatrix::from([[1.0, 2.0], [3.0, 4.0]]);

	let (values, vectors) = m.eigen();
	let inv_vectors = vectors.inverse();

	let reconstructed =
		vectors * RowMatrix::from_diagonal(values) * inv_vectors;

	for i in 0..2 {
		for j in 0..2 {
			assert_almost_eq!(reconstructed[(i, j)], m[(i, j)]);
		}
	}
}
