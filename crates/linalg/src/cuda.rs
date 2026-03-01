use cudarc::driver::safe::{DeviceRepr, ValidAsZeroBits};

use crate::{RowMatrix, Vector};

// SAFETY: Vector is repr(C) over an array
unsafe impl<T, const N: usize> DeviceRepr for Vector<T, N> where T: DeviceRepr {}
// SAFETY: Matrix is repr(C) over an array of arrays
unsafe impl<T, const N: usize, const M: usize> DeviceRepr for RowMatrix<T, N, M> where
	T: DeviceRepr
{
}
// SAFETY: see `bytemuck` `Zeroable` implementation guarantees
unsafe impl<T, const N: usize> ValidAsZeroBits for Vector<T, N> where
	T: ValidAsZeroBits
{
}
// SAFETY: see `bytemuck` `Zeroable` implementation guarantees
unsafe impl<T, const N: usize, const M: usize> ValidAsZeroBits
	for RowMatrix<T, N, M>
where
	T: ValidAsZeroBits,
{
}
