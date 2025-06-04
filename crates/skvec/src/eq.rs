use std::cmp::PartialEq;

use crate::SkVec;

macro_rules! impl_eq {
	($this:ty, $other:ty $(, $($extra:tt)*)?) => {
		impl<T: PartialEq $(, $($extra)*)?> PartialEq<$other> for $this {
			fn eq(&self, other: &$other) -> bool {
				if self.len() != other.len() {
					return false;
				}

				for (a, b) in self.iter().zip(other.iter()) {
					if a != b {
						return false;
					}
				}

				true
			}
		}
	};
}

impl_eq!(SkVec<T>, SkVec<T>);

impl_eq!(SkVec<T>, Vec<T>);
impl_eq!(SkVec<T>, [T]);
impl_eq!(SkVec<T>, &[T]);
impl_eq!(SkVec<T>, [T; N], const N: usize);

impl_eq!(Vec<T>, SkVec<T>);
impl_eq!([T], SkVec<T>);
impl_eq!(&[T], SkVec<T>);
impl_eq!([T; N], SkVec<T>, const N: usize);

impl<T: Eq> Eq for SkVec<T> {}
