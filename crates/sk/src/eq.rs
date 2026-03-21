use std::cmp::PartialEq;

use crate::SkBuf;

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

impl_eq!(SkBuf<T>, SkBuf<T>);

impl_eq!(SkBuf<T>, Vec<T>);
impl_eq!(SkBuf<T>, [T]);
impl_eq!(SkBuf<T>, &[T]);
impl_eq!(SkBuf<T>, [T; N], const N: usize);

impl_eq!(Vec<T>, SkBuf<T>);
impl_eq!([T], SkBuf<T>);
impl_eq!(&[T], SkBuf<T>);
impl_eq!([T; N], SkBuf<T>, const N: usize);

impl<T: Eq> Eq for SkBuf<T> {}
