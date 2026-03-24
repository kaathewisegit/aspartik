use arbitrary::{Result, Unstructured};

use crate::{Vector, matrix::Matrix};

// TODO: dynamic bounds
fn small_float(u: &mut Unstructured) -> Result<f64> {
	let integer: u64 = u.arbitrary()?;
	let out = 1.0 / (1u64 << 54) as f64 * (integer >> 10) as f64;
	Ok(out)
}

pub fn vector<const N: usize>(u: &mut Unstructured) -> Result<[f64; N]> {
	let mut out = <[f64; N]>::from_element(0.0);

	for element in out.iter_mut() {
		if let Ok(new) = small_float(u) {
			*element = new;
		} else {
			// leave the rest as zeros
			break;
		}
	}

	Ok(out)
}

pub fn matrix<const N: usize, const M: usize>(
	u: &mut Unstructured,
) -> Result<[[f64; M]; N]> {
	let mut out: [[f64; M]; N] = Matrix::zeros();

	for row in out.iter_mut() {
		for element in row.iter_mut() {
			if let Ok(new) = small_float(u) {
				*element = new;
			} else {
				// leave the rest as zeros
				break;
			}
		}
	}

	Ok(out)
}

pub fn symmetric<const N: usize>(
	u: &mut Unstructured,
) -> Result<[[f64; N]; N]> {
	let mut out: [[f64; N]; N] = Matrix::zeros();

	#[expect(clippy::needless_range_loop)]
	for i in 0..N {
		for j in i..N {
			if let Ok(new) = small_float(u) {
				out[i][j] = new;
				out[j][i] = new;
			} else {
				// leave the rest as zeros
				break;
			}
		}
	}

	Ok(out)
}
