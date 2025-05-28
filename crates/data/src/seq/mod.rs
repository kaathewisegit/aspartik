use anyhow::{Context, Error, Result};

use std::{
	fmt,
	ops::{Deref, DerefMut},
};

use crate::nucleotides::DnaNucleotide;

#[cfg(feature = "python")]
pub mod python;

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

// XXX: move to a generic sequence-parsing utility function, so that it can be
// shared between sequence types
fn highlight_error(src: &str, index: usize) -> String {
	const MAX_WIDTH: usize = 60;
	if src.len() > MAX_WIDTH {
		let mut out = String::from(
			"Illegal character encountered in the sequence:\n> ",
		);
		let mut padding = 2;

		let start = if index > 40 {
			out.push_str("...");
			padding += 3;
			index - 40
		} else {
			0
		};

		let end = std::cmp::min(start + MAX_WIDTH, src.len());
		out.push_str(&src[start..end]);
		if end < src.len() {
			out.push_str("...");
		}
		out.push('\n');
		for _ in 0..(padding + index - start) {
			out.push(' ');
		}
		out.push('^');

		out
	} else {
		format!(
			"Illegal character encountered in the sequence:\n> {}\n  {:index$}^",
			src,
			"",
		)
	}
}

// XXX: see above
impl<C: Character> TryFrom<&str> for Seq<C> {
	type Error = Error;

	fn try_from(value: &str) -> Result<Self> {
		let mut chars = Vec::with_capacity(value.len());

		for ch in value.chars() {
			let character = ch.try_into().with_context(|| {
				highlight_error(value, chars.len())
			})?;
			chars.push(character);
		}

		Ok(chars.into())
	}
}

// Character-agnostic methods
impl<C: Character> Seq<C> {
	/// Reverses the characters in-place.
	pub fn reverse(&mut self) {
		self.inner.reverse();
	}

	/// Returns the character slice which backs the sequence.  Mutating it
	/// will change the sequence accordingly.
	pub fn as_mut_slice(&mut self) -> &mut [C] {
		&mut self.inner
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

#[cfg(test)]
mod test {
	use super::*;
	use DnaNucleotide::*;

	#[test]
	fn decode() {
		let s = "ACTGxACTG";
		let seq: Result<Seq<DnaNucleotide>> = s.try_into();
		assert!(seq.is_err());
	}

	#[test]
	fn count() {
		let s: DnaSeq = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC".try_into().unwrap();

		assert_eq!(s.count(DnaNucleotide::Adenine), 20);
		assert_eq!(s.count(DnaNucleotide::Cytosine), 12);
		assert_eq!(s.count(DnaNucleotide::Guanine), 17);
		assert_eq!(s.count(DnaNucleotide::Thymine), 21);
	}

	#[test]
	fn dna_complement() {
		let s: DnaSeq = "AAAACCCGGT".try_into().unwrap();

		assert_eq!(s.reverse_complement().to_string(), "ACCGGGTTTT");
	}

	#[test]
	fn hamming() {
		let s1: DnaSeq = "GAGCCTACTAACGGGAT".try_into().unwrap();
		let s2: DnaSeq = "CATCGTAATGACGGCCT".try_into().unwrap();

		assert_eq!(s1.hamming_distance(s2), 7);
	}

	#[test]
	fn index() {
		let mut s = DnaSeq::try_from("ACGT").unwrap();
		assert_eq!(s[0], Adenine);
		assert_eq!(s[1], Cytosine);
		assert_eq!(s[2], Guanine);
		assert_eq!(s[3], Thymine);

		s[0] = Thymine;
		s[1] = Cytosine;
		s[2] = Guanine;
		s[3] = Adenine;
		assert_eq!(s[0], Thymine);
		assert_eq!(s[1], Cytosine);
		assert_eq!(s[2], Guanine);
		assert_eq!(s[3], Adenine);
	}

	#[test]
	fn iter() {
		let s = DnaSeq::try_from("GAGCCT").unwrap();
		let mut iter = s.iter().copied();
		assert_eq!(iter.next(), Some(Guanine));
		assert_eq!(iter.next(), Some(Adenine));
		assert_eq!(iter.next(), Some(Guanine));
		assert_eq!(iter.next(), Some(Cytosine));
		assert_eq!(iter.next(), Some(Cytosine));
		assert_eq!(iter.next(), Some(Thymine));
	}
}
