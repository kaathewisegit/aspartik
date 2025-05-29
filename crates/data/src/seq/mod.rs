use anyhow::Error;

use std::{
	fmt,
	ops::{Deref, DerefMut},
};

use crate::nucleotides::DnaNucleotide;

pub mod distance;
mod parse;
#[cfg(feature = "python")]
pub mod python;

pub use parse::{parse_bytes, parse_str};

/// A character in a sequence alphabet.
///
/// # Safety
///
/// The type must have the same size an alignment as `u8`, so that `[T]` can be
/// casted to `[u8]`.  In practice this means that the size of the type must be
/// one byte and there are no alignment requirements (all types are 1-byte
/// aligned).
pub unsafe trait Character:
	TryFrom<u8, Error = Error>
	+ TryFrom<char, Error = Error>
	+ Into<u8>
	+ Into<char>
	+ Copy
	+ Eq
{
}

// DnaNucleotide is `repr(u8)`.
unsafe impl Character for DnaNucleotide {}

pub trait Seq {
	type Character: Character;

	fn as_slice(&self) -> &[Self::Character];

	fn fmt_impl(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use fmt::Write;

		for character in self.as_slice().iter().copied() {
			f.write_char(character.into())?;
		}
		Ok(())
	}

	fn as_bytes(&self) -> &[u8] {
		let slice = self.as_slice();
		// SAFETY: `Character` must be equivalent to a byte
		unsafe {
			std::mem::transmute::<&[Self::Character], &[u8]>(slice)
		}
	}

	fn iter(&self) -> std::slice::Iter<'_, Self::Character> {
		self.as_slice().iter()
	}

	fn len(&self) -> usize {
		self.as_slice().len()
	}

	fn is_empty(&self) -> bool {
		self.as_slice().is_empty()
	}

	/// Counts how many times the character `c` occurs in the sequence.
	fn count(&self, c: Self::Character) -> usize {
		let mut out = 0;

		for current in self.iter().copied() {
			if current == c {
				out += 1
			}
		}

		out
	}
}

pub trait FromChars: Seq {
	fn from_vec(chars: Vec<Self::Character>) -> Self;

	fn from_slice(chars: &[Self::Character]) -> Self
	where
		Self: Sized,
	{
		Self::from_vec(chars.to_vec())
	}
}

pub trait SeqMut: Seq + AsMut<[Self::Character]> {
	fn push(&mut self, ch: Self::Character);

	fn extend<S>(&mut self, other: &S)
	where
		S: Seq<Character = Self::Character>;

	/// Reverses the characters in-place.
	fn reverse(&mut self) {
		self.as_mut().reverse();
	}
}

impl<C: Character> Seq for &[C] {
	type Character = C;

	fn as_slice(&self) -> &[C] {
		self
	}
}

impl<C: Character, const N: usize> Seq for [C; N] {
	type Character = C;

	fn as_slice(&self) -> &[C] {
		self
	}
}

impl<C: Character> Seq for Vec<C> {
	type Character = C;

	fn as_slice(&self) -> &[C] {
		self.as_slice()
	}
}

impl<C: Character> FromChars for Vec<C> {
	fn from_vec(chars: Vec<C>) -> Self {
		chars
	}
}

impl<C: Character> SeqMut for Vec<C> {
	fn push(&mut self, ch: C) {
		self.push(ch)
	}

	fn extend<S>(&mut self, other: &S)
	where
		S: Seq<Character = C>,
	{
		self.extend_from_slice(other.as_slice())
	}
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SeqBuf<C: Character> {
	inner: Box<[C]>,
}

impl<C: Character> Seq for SeqBuf<C> {
	type Character = C;

	fn as_slice(&self) -> &[C] {
		&self.inner
	}
}

impl<C: Character> FromChars for SeqBuf<C> {
	fn from_vec(chars: Vec<C>) -> Self {
		Self {
			inner: chars.into_boxed_slice(),
		}
	}
}

impl<C: Character> Deref for SeqBuf<C> {
	type Target = [C];

	fn deref(&self) -> &[C] {
		&self.inner
	}
}

impl<C: Character> DerefMut for SeqBuf<C> {
	fn deref_mut(&mut self) -> &mut [C] {
		&mut self.inner
	}
}

impl<C: Character> fmt::Display for SeqBuf<C> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.fmt_impl(f)
	}
}

pub type DnaSeq = SeqBuf<DnaNucleotide>;

// DNA-specific methods
impl DnaSeq {
	/// Returns the sequence complement of `self`.  Note that this function
	/// doesn't reverse the direction of the sequence, use
	/// [`reverse_complement`][`DnaSeq::reverse_complement`] for that.
	pub fn complement(&self) -> Self {
		let mut out = self.clone();
		for base in out.inner.iter_mut() {
			*base = base.complement();
		}
		out
	}

	pub fn reverse_complement(&self) -> Self {
		let mut out = self.complement();
		out.reverse();

		out
	}
}
