use super::RowMatrix;
use bytemuck::{Pod, Zeroable};

// SAFETY: `RowMatrix` stores a contiguous array of `T`'s without padding, so if
// `T: Zeroable`, so is `RowMatrix<T, _>`.
unsafe impl<T, const N: usize, const M: usize> Zeroable for RowMatrix<T, N, M> where
	T: Zeroable
{
}
// SAFETY:
// - Is inhabited (`RowMatrix<T, 0>` has a single value, similar to `()`)
// - Has no padding
// - All fields are `T`, and `T: Pod`
// - Is `repr(C)`
// - Can't have interior mutability because `T: Pod`
unsafe impl<T, const N: usize, const M: usize> Pod for RowMatrix<T, N, M> where
	T: Pod
{
}
