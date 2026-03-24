use num_traits::Num;

use crate::vector::Vector;

pub trait Matrix<T, const N: usize, const M: usize>: Sized
where
	T: Copy + Num,
{
	fn at(&self, row: usize, column: usize) -> &T;

	fn at_mut(&mut self, row: usize, column: usize) -> &mut T;

	fn from_element(value: T) -> Self;

	fn for_each<F>(&mut self, f: F)
	where
		F: Fn(&mut T);

	fn zeros() -> Self {
		Self::from_element(T::zero())
	}

	fn mul<const P: usize>(self, rhs: impl Matrix<T, M, P>) -> Self {
		let mut out = Self::zeros();

		for i in 0..N {
			for j in 0..P {
				for k in 0..M {
					*out.at_mut(i, j) = *out.at(i, j)
						+ *self.at(i, k)
							* *rhs.at(k, j);
				}
			}
		}

		out
	}

	// fn mul_scalar(self, rhs: T) -> Self {
	// 	let mut out = self;

	// 	for i in 0..N {
	// 		for j in 0..M {
	// 			out[(i, j)] *= rhs;
	// 		}
	// 	}

	// 	out
	// }
}

pub trait SquareMatrix<T, const N: usize>: Matrix<T, N, N>
where
	T: Copy + Num,
{
	fn from_diagonal(diag: impl Vector<T, N>) -> Self {
		let mut out = Self::from_element(T::zero());
		for i in 0..N {
			*out.at_mut(i, i) = diag[i];
		}
		out
	}

	fn identity() -> Self {
		Self::from_diagonal([T::one(); N])
	}

	fn trace(&self) -> T {
		let mut out = *self.at(0, 0);
		for i in 1..N {
			out = out + *self.at(i, i);
		}
		out
	}
}

impl<T, const N: usize, M> SquareMatrix<T, N> for M
where
	T: Copy + Num,
	M: Matrix<T, N, N>,
{
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M> for [[T; M]; N]
where
	T: Copy + Num,
{
	fn at(&self, row: usize, column: usize) -> &T {
		&self[row][column]
	}

	fn at_mut(&mut self, row: usize, column: usize) -> &mut T {
		&mut self[row][column]
	}

	fn from_element(value: T) -> Self {
		[[value; M]; N]
	}

	fn for_each<F>(&mut self, f: F)
	where
		F: Fn(&mut T),
	{
		for i in 0..N {
			for j in 0..M {
				f(self.at_mut(i, j));
			}
		}
	}
}

pub fn transpose<T: Copy + Num, const N: usize, const M: usize, S, D>(
	src: &S,
) -> D
where
	S: Matrix<T, N, M>,
	D: Matrix<T, M, N>,
{
	let mut out = D::zeros();

	for i in 0..N {
		for j in 0..M {
			*out.at_mut(j, i) = *src.at(i, j);
		}
	}

	out
}
