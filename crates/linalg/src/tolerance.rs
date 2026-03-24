use math::tolerance::Tolerance;
use num_traits::Zero;

use crate::RowMatrix;

impl<T: Tolerance, const N: usize, const M: usize> Tolerance
	for RowMatrix<T, N, M>
where
	T: Tolerance + Copy,
	T::Diff: Zero + PartialOrd,
	T::Relative: Zero + PartialOrd,
{
	type Diff = T::Diff;
	type Relative = T::Relative;

	fn abs_diff(&self, other: &Self) -> T::Diff {
		let mut max_diff = T::Diff::zero();

		for i in 0..N {
			for j in 0..M {
				let diff = self[i][j].abs_diff(&other[i][j]);
				if diff > max_diff {
					max_diff = diff;
				}
			}
		}

		max_diff
	}

	fn relative(&self, other: &Self) -> T::Relative {
		let mut max_rel = T::Relative::zero();

		for i in 0..N {
			for j in 0..M {
				let rel = self[i][j].relative(&other[i][j]);
				if rel > max_rel {
					max_rel = rel;
				}
			}
		}

		max_rel
	}

	fn ulps(&self, other: &Self) -> u64 {
		let mut max_ulps = 0;

		for i in 0..N {
			for j in 0..M {
				let ulps = self[i][j].ulps(&other[i][j]);
				if ulps > max_ulps {
					max_ulps = ulps;
				}
			}
		}

		max_ulps
	}
}
