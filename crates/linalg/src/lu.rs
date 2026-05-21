use bytemuck::Zeroable;
use num_traits::Float;

use crate::{MatrixMut, MatrixRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LuError {
	Singular,
}

#[derive(Debug, Clone)]
pub struct LU<T: Float> {
	lu: Box<[T]>,
	piv: Box<[usize]>,
	#[expect(dead_code)]
	sign_negative: bool,
}

impl<T> LU<T>
where
	T: Float + Zeroable,
{
	pub fn factor(a: MatrixRef<T>) -> Result<Self, LuError> {
		assert!(a.is_square());
		let n = a.num_rows();

		let mut lu_box = a.to_boxed_slice();
		let mut piv = vec![0usize; n].into_boxed_slice();
		let mut sign_negative = false;

		let mut max_abs_val = T::zero();
		for value in &mut lu_box {
			let value = value.abs();
			if value > max_abs_val {
				max_abs_val = value;
			}
		}
		let eps = T::epsilon();
		let tol = (eps * max_abs_val * T::from(n).unwrap())
			.max(T::min_positive_value());

		let mut lu = MatrixMut::from_slice(&mut lu_box, n, n);

		for k in 0..n {
			let mut p = k;
			let mut max_col = lu[(k, k)].abs();

			for i in k + 1..n {
				let v = lu[(i, k)].abs();
				if v > max_col {
					max_col = v;
					p = i;
				}
			}

			if max_col < tol {
				return Err(LuError::Singular);
			}

			piv[k] = p;
			if p != k {
				lu.reborrow().swap_rows(k, p);
				sign_negative = !sign_negative;
			}

			let pivot_val = lu[(k, k)];
			for i in k + 1..n {
				lu[(i, k)] = lu[(i, k)] / pivot_val;
				let factor = lu[(i, k)];
				for j in k + 1..n {
					let row_k_j = lu[(k, j)];
					lu[(i, j)] =
						lu[(i, j)] - factor * row_k_j;
				}
			}
		}

		Ok(LU {
			lu: lu_box,
			piv,
			sign_negative,
		})
	}

	pub fn inverse(&self, mut dst: MatrixMut<T>) {
		dst.reborrow().identity();
		let n = self.piv.len();

		let lu = MatrixRef::from_slice(&self.lu, n, n);

		// permutation
		for k in 0..n {
			dst.reborrow().swap_rows(k, self.piv[k]);
		}

		// forward substitution
		for i in 0..n {
			for j in 0..i {
				let factor = lu[(i, j)];
				for col in 0..n {
					dst[(i, col)] = dst[(i, col)]
						- factor * dst[(j, col)];
				}
			}
		}

		// backward substitution
		for i in (0..n).rev() {
			for j in i + 1..n {
				let factor = lu[(i, j)];
				for col in 0..n {
					dst[(i, col)] = dst[(i, col)]
						- factor * dst[(j, col)];
				}
			}
			let diag = lu[(i, i)];
			for col in 0..n {
				dst[(i, col)] = dst[(i, col)] / diag;
			}
		}
	}
}

pub fn inverse<'a, T, I, O>(m: I, dst: O)
where
	T: Float + Zeroable + 'a,
	I: Into<MatrixRef<'a, T>>,
	O: Into<MatrixMut<'a, T>>,
{
	let lu = LU::factor(m.into()).unwrap();
	lu.inverse(dst.into())
}
