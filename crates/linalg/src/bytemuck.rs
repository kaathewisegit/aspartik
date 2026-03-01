use super::{RowMatrix, Vector};
use bytemuck::{Pod, Zeroable};

// SAFETY: `Vector` stores a contiguous array of `T`'s without padding, so if
// `T: Zeroable`, so is `Vector<T, _>`.
unsafe impl<T, const N: usize> Zeroable for Vector<T, N> where T: Zeroable {}
// SAFETY:
// - Is inhabited (`Vector<T, 0>` has a single value, similar to `()`)
// - Has no padding
// - All fields are `T`, and `T: Pod`
// - Is `repr(C)`
// - Can't have interior mutability because `T: Pod`
unsafe impl<T, const N: usize> Pod for Vector<T, N> where T: Pod {}

// SAFETY: same as `Vector<T>`
unsafe impl<T, const N: usize, const M: usize> Zeroable for RowMatrix<T, N, M> where
	T: Zeroable
{
}
// SAFETY: same as `Vector<T>`
unsafe impl<T, const N: usize, const M: usize> Pod for RowMatrix<T, N, M> where
	T: Pod
{
}
