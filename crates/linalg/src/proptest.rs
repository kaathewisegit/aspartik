use proptest::prelude::*;

use std::ops::RangeInclusive;

use crate::{RowMatrix, Vector};

const PROPTEST_F64: RangeInclusive<f64> = -1000.0..=1000.0;

pub fn vector<const N: usize>() -> impl Strategy<Value = Vector<f64, N>> + Clone
{
	[PROPTEST_F64; N].prop_map_into()
}

pub fn matrix<const N: usize, const M: usize>()
-> impl Strategy<Value = RowMatrix<f64, N, M>> {
	vec![vector::<N>(); M].prop_map(|val| {
		let arr = val.as_slice().first_chunk::<M>().unwrap();
		(*arr).into()
	})
}

pub fn symmetric<const N: usize>() -> impl Strategy<Value = RowMatrix<f64, N, N>>
{
	matrix().prop_map(|mut matrix| {
		for i in 1..N {
			for j in 0..i {
				matrix[i][j] = matrix[j][i]
			}
		}
		assert!(matrix.is_symmetric());
		matrix
	})
}
