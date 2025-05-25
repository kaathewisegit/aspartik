use cudarc::driver::safe::{DeviceRepr, ValidAsZeroBits};

use crate::{RowMatrix, Vector};

// SAFETY: Vector and RowMatrix are repr(C)
unsafe impl<T, const N: usize> DeviceRepr for Vector<T, N> where
	T: DeviceRepr + Copy
{
}
unsafe impl<T, const N: usize, const M: usize> DeviceRepr for RowMatrix<T, N, M> where
	T: DeviceRepr + Copy
{
}
unsafe impl<T, const N: usize> ValidAsZeroBits for Vector<T, N> where
	T: ValidAsZeroBits + Copy
{
}
unsafe impl<T, const N: usize, const M: usize> ValidAsZeroBits
	for RowMatrix<T, N, M>
where
	T: ValidAsZeroBits + Copy,
{
}
