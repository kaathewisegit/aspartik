use std::cmp::PartialEq;

use crate::SkVec;

macro_rules! impl_inside {
	($self:ident, $other:ident) => {{
		if $self.len() != $other.len() {
			return false;
		}

		for (a, b) in $self.iter().zip($other.iter()) {
			if a != b {
				return false;
			}
		}

		true
	}};
}

impl<T: PartialEq> PartialEq for SkVec<T> {
	fn eq(&self, other: &Self) -> bool {
		impl_inside!(self, other)
	}
}

impl<T: PartialEq> PartialEq<[T]> for SkVec<T> {
	fn eq(&self, other: &[T]) -> bool {
		impl_inside!(self, other)
	}
}

impl<T: PartialEq, const N: usize> PartialEq<[T; N]> for SkVec<T> {
	fn eq(&self, other: &[T; N]) -> bool {
		impl_inside!(self, other)
	}
}

impl<T: PartialEq> PartialEq<Vec<T>> for SkVec<T> {
	fn eq(&self, other: &Vec<T>) -> bool {
		impl_inside!(self, other)
	}
}

impl<T: PartialEq> PartialEq<SkVec<T>> for [T] {
	fn eq(&self, other: &SkVec<T>) -> bool {
		impl_inside!(self, other)
	}
}

impl<T: PartialEq, const N: usize> PartialEq<SkVec<T>> for [T; N] {
	fn eq(&self, other: &SkVec<T>) -> bool {
		impl_inside!(self, other)
	}
}

impl<T: PartialEq> PartialEq<SkVec<T>> for Vec<T> {
	fn eq(&self, other: &SkVec<T>) -> bool {
		impl_inside!(self, other)
	}
}
