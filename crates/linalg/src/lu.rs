use num_traits::Float;

use std::marker::PhantomData;

use crate::{ConstSquareMatrix, Vector};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LuError {
	Singular,
}

#[derive(Debug, Clone)]
pub struct LU<T, M, const N: usize>
where
	T: Float,
	M: ConstSquareMatrix<T, N>,
{
	pub(crate) lu: M,
	pub(crate) piv: [usize; N],
	pub(crate) sign_negative: bool,
	t: PhantomData<T>,
}

impl<T, M, const N: usize> LU<T, M, N>
where
	T: Float,
	M: ConstSquareMatrix<T, N>,
{
	pub fn factor(
		a: &impl ConstSquareMatrix<T, N>,
	) -> Result<Self, LuError> {
		let mut lu: M = a.convert();
		let mut piv = [0usize; N];
		let mut sign_negative = false;

		let mut max_abs_val = T::zero();
		lu.for_each(|value| {
			let value = value.abs();
			if value > max_abs_val {
				max_abs_val = value;
			}
		});
		let eps = T::epsilon();
		let tol = (eps * max_abs_val * T::from(N).unwrap())
			.max(T::min_positive_value());

		#[expect(clippy::needless_range_loop)]
		for k in 0..N {
			let mut p = k;
			let mut max_col = lu.at(k, k).abs();

			for i in k + 1..N {
				let v = lu.at(i, k).abs();
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
				lu.swap_rows(k, p);
				sign_negative = !sign_negative;
			}

			let pivot_val = *lu.at(k, k);
			for i in k + 1..N {
				*lu.at_mut(i, k) = *lu.at(i, k) / pivot_val;
				let factor = *lu.at(i, k);
				for j in k + 1..N {
					let row_k_j = *lu.at(k, j);
					*lu.at_mut(i, j) =
						*lu.at(i, j) - factor * row_k_j;
				}
			}
		}

		Ok(LU {
			lu,
			piv,
			sign_negative,
			t: PhantomData,
		})
	}

	pub fn solve<O>(&self, b: &impl Vector<T, N>) -> O
	where
		O: Vector<T, N>,
	{
		let mut x = b.clone();

		for i in 0..N {
			x.swap(i, self.piv[i]);
		}

		for i in 0..N {
			for j in 0..i {
				x[i] = x[i] - *self.lu.at(i, j) * x[j];
			}
		}

		for i in (0..N).rev() {
			for j in i + 1..N {
				x[i] = x[i] - *self.lu.at(i, j) * x[j];
			}
			x[i] = x[i] / *self.lu.at(i, i);
		}

		x.convert()
	}

	pub fn det(&self) -> T {
		let mut det = if self.sign_negative {
			-T::one()
		} else {
			T::one()
		};
		for i in 0..N {
			det = det * *self.lu.at(i, i);
		}
		det
	}

	pub fn inverse(&self) -> M {
		let mut inv = M::identity();

		// permutation
		for k in 0..N {
			inv.swap_rows(k, self.piv[k]);
		}

		// forward substitution
		for i in 0..N {
			for j in 0..i {
				let factor = *self.lu.at(i, j);
				for col in 0..N {
					*inv.at_mut(i, col) = *inv.at(i, col)
						- factor * *inv.at(j, col);
				}
			}
		}

		// backward substitution
		for i in (0..N).rev() {
			for j in i + 1..N {
				let factor = *self.lu.at(i, j);
				for col in 0..N {
					*inv.at_mut(i, col) = *inv.at(i, col)
						- factor * *inv.at(j, col);
				}
			}
			let diag = *self.lu.at(i, i);
			for col in 0..N {
				*inv.at_mut(i, col) = *inv.at(i, col) / diag;
			}
		}

		inv
	}
}

pub fn inverse<const N: usize, T: Float, I, O>(m: &I) -> O
where
	I: ConstSquareMatrix<T, N>,
	O: ConstSquareMatrix<T, N>,
{
	let lu = LU::<T, O, N>::factor(m).unwrap();
	lu.inverse()
}
