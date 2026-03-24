use cudarc::driver::safe::{DeviceRepr, ValidAsZeroBits};

use crate::RowMatrix;

// SAFETY: Matrix is repr(C) over an array of arrays
unsafe impl<T, const N: usize, const M: usize> DeviceRepr for RowMatrix<T, N, M> where
	T: DeviceRepr
{
}
// SAFETY: see `bytemuck` `Zeroable` implementation guarantees
unsafe impl<T, const N: usize, const M: usize> ValidAsZeroBits
	for RowMatrix<T, N, M>
where
	T: ValidAsZeroBits,
{
}
