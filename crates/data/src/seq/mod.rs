use anyhow::Error;

use std::{
	fmt,
	ops::{Deref, DerefMut},
};

use crate::nucleotides::DnaNucleotide;

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

pub trait SeqView:
	for<'a> From<&'a [Self::Character]>
	+ From<Vec<Self::Character>>
	+ AsRef<[Self::Character]>
{
	type Character: Character;

	fn fmt_impl(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use fmt::Write;

		for character in self.as_ref().iter().copied() {
			f.write_char(character.into())?;
		}
		Ok(())
	}

	fn as_bytes(&self) -> &[u8] {
		let slice = self.as_ref();
		// SAFETY: `Character` must be equivalent to a byte
		unsafe {
			std::mem::transmute::<&[Self::Character], &[u8]>(slice)
		}
	}

	fn iter(&self) -> std::slice::Iter<'_, Self::Character> {
		self.as_ref().iter()
	}

	fn len(&self) -> usize {
		self.as_ref().len()
	}

	fn is_empty(&self) -> bool {
		self.as_ref().is_empty()
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

	/// Calculates the Hamming distance between two sequences.
	///
	///
	/// # Panics
	///
	/// Panics if lengths of the sequences are not equal.
	fn hamming_distance<S>(&self, other: S) -> usize
	where
		S: SeqView<Character = Self::Character>,
	{
		assert_eq!(self.len(), other.len());

		let mut out = 0;

		for (a, b) in self.as_ref().iter().zip(other.as_ref()) {
			if a != b {
				out += 1;
			}
		}

		out
	}
}

pub trait SeqMutView: SeqView + AsMut<[Self::Character]> {
	fn push(&mut self, ch: Self::Character);

	fn extend<S: SeqView>(&mut self, other: &S);

	fn append<S: SeqMutView>(&mut self, other: &mut S);

	/// Reverses the characters in-place.
	fn reverse(&mut self) {
		self.as_mut().reverse();
	}
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Seq<C: Character> {
	inner: Box<[C]>,
}

impl<C: Character> SeqView for Seq<C> {
	type Character = C;
}

impl<C: Character> Deref for Seq<C> {
	type Target = [C];

	fn deref(&self) -> &[C] {
		&self.inner
	}
}

impl<C: Character> DerefMut for Seq<C> {
	fn deref_mut(&mut self) -> &mut [C] {
		&mut self.inner
	}
}

impl<C: Character> AsRef<[C]> for Seq<C> {
	fn as_ref(&self) -> &[C] {
		self
	}
}

impl<C: Character> fmt::Display for Seq<C> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.fmt_impl(f)
	}
}

impl<C: Character> From<&[C]> for Seq<C> {
	fn from(value: &[C]) -> Self {
		Seq {
			inner: value.into(),
		}
	}
}

impl<C: Character> From<Vec<C>> for Seq<C> {
	fn from(value: Vec<C>) -> Self {
		Self {
			inner: value.into_boxed_slice(),
		}
	}
}

pub type DnaSeq = Seq<DnaNucleotide>;

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
