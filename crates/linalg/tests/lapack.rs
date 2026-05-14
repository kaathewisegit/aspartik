use arbitrary::{Result, Unstructured};
use arbtest::arbtest;

use linalg::{
	arbitrary::{normalized_array, symmetric},
	eigen,
	lapack::{self, inverse},
	matrix::{Matrix, SquareMatrix},
};
use math::assert_almost_eq;

type M<const N: usize> = [[f64; N]; N];

fn reconstruct<const N: usize>(m: M<N>, relative: f64) {
	let decomp = eigen(&m);
	let (values, vectors) = (decomp.eigenvalues, decomp.eigenvectors);
	let inv_vectors: M<N> = inverse(&vectors);

	let diag: M<N> = SquareMatrix::from_diagonal(values);

	let reconstructed: M<N> =
		vectors.mul::<_, M<N>>(&diag).mul(&inv_vectors);

	for i in 0..N {
		for j in 0..N {
			assert_almost_eq!(
				*reconstructed.at(i, j),
				*m.at(i, j),
				relative = relative,
			);
		}
	}
}

#[test]
fn reconstruction_basic() {
	let m = [[1.0, 2.0], [3.0, 4.0]];
	reconstruct(m, 1e-10);
}

#[test]
fn reconstruction_symmetric() {
	arbtest(|u| {
		let m = symmetric::<4>(u)?;
		reconstruct(m, 1e-9);
		Ok(())
	})
	.size_min(128)
	.budget_ms(500);
}

#[test]
fn simple_complex_eigenvalues() {
	let m = [[0.0, -1.0], [1.0, 0.0]];
	let decomp = eigen(&m);
	let (re, im) = (decomp.eigenvalues, decomp.eigenvalues_img);

	assert_almost_eq!(re[0], 0.0);
	assert_almost_eq!(re[1], 0.0);
	assert_almost_eq!(im[0], 1.0);
	assert_almost_eq!(im[1], -1.0);
}

fn gtr(u: &mut Unstructured) -> Result<M<4>> {
	let [p_a, p_c, p_g, p_t] = normalized_array::<4>(u)?;
	let [a, b, c, d, e, f] = normalized_array::<6>(u)?;

	let mut gtr = [
		[-a * p_c - b * p_g - c * p_t, a * p_c, b * p_g, c * p_t],
		[a * p_a, -a * p_a - d * p_g - e * p_t, d * p_g, e * p_t],
		[b * p_a, d * p_c, -b * p_a - d * p_c - f * p_t, f * p_t],
		[c * p_a, e * p_c, f * p_g, -c * p_a - e * p_c - f * p_g],
	];
	let div = 2.0
		* (a * p_a * p_c
			+ b * p_a * p_g + c * p_a * p_t
			+ d * p_c * p_g + e * p_c * p_t
			+ f * p_g * p_t);
	gtr.for_each(|e| *e /= div);
	Ok(gtr)
}

#[test]
fn compare() {
	arbtest(|u| {
		let gtr = gtr(u)?;
		let (mut lapack_values, _) = lapack::eigen(&gtr);
		let decomp = eigen(&gtr);
		let mut our_values = decomp.eigenvalues;

		lapack_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
		our_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

		// TODO: compare vectors too

		for i in 0..4 {
			assert_almost_eq!(
				lapack_values[i],
				our_values[i],
				epsilon = 1e-13,
				relative = 1e-13
			);
		}

		Ok(())
	})
	.size_min(128)
	.budget_ms(500);
}
