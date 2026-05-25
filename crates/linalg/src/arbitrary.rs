use arbitrary::{Result, Unstructured};

// TODO: dynamic bounds
fn small_float(u: &mut Unstructured) -> Result<f64> {
	let integer: u64 = u.arbitrary()?;
	let out = 1.0 / (1u64 << 54) as f64 * (integer >> 10) as f64;
	Ok(out)
}

pub fn matrix<const N: usize, const M: usize>(
	u: &mut Unstructured,
) -> Result<[[f64; M]; N]> {
	let mut out: [[f64; M]; N] = [[0.0; M]; N];

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
	let mut out: [[f64; N]; N] = [[0.0; N]; N];

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

pub fn normalized_array<const N: usize>(
	u: &mut Unstructured,
) -> Result<[f64; N]> {
	let mut out = [0.0; N];

	for value in &mut out {
		*value = f64::from(u.arbitrary::<u32>()?);
	}
	let sum: f64 = out.iter().sum();

	for value in &mut out {
		*value /= sum;
	}

	Ok(out)
}
