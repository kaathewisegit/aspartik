use bytes::{BufMut, Bytes, BytesMut};

use std::{
	fmt,
	marker::PhantomData,
	mem,
	ops::{Deref, DerefMut, RangeBounds},
	slice,
};

use crate::nucleotides::DnaNucleotide;

pub mod distance;
mod parse;
#[cfg(feature = "python")]
pub mod python;

pub use parse::{parse_append_bytes, parse_append_str, parse_bytes, parse_str};

/// A character in a sequence alphabet.
///
/// # Safety
///
/// The type must have the same size an alignment as `u8`, so that `[T]` can be
/// casted to `[u8]`.  In practice this means that the size of the type must be
/// one byte and there are no alignment requirements (all types are 1-byte
/// aligned).
pub unsafe trait Character: Copy + Eq {
	fn from_ascii(char: u8) -> Option<Self>;

	fn to_ascii(&self) -> u8;

	fn from_byte(b: u8) -> Option<Self>;

	fn to_byte(&self) -> u8;

	fn into_byte(self) -> u8;
}

fn c2b<C: Character>(characters: &[C]) -> &[u8] {
	let ptr = characters.as_ptr() as *const u8;
	// SAFETY: characters must be equal in layout to `u8` bytes
	unsafe { slice::from_raw_parts(ptr, characters.len()) }
}

/// Cast a slice of bytes to a slice of characters
///
/// # Safety
///
/// All bytes in `bytes` must be valid characters.
unsafe fn b2c<C: Character>(bytes: &[u8]) -> &[C] {
	let ptr = bytes.as_ptr() as *const C;
	// SAFETY: characters must be equal in layout to `u8` bytes
	unsafe { slice::from_raw_parts(ptr, bytes.len()) }
}

/// Cast a mutable slice of bytes to a slice of characters
///
/// # Safety
///
/// All bytes in `bytes` must be valid characters.
unsafe fn b2c_mut<C: Character>(bytes: &mut [u8]) -> &mut [C] {
	let ptr = bytes.as_ptr() as *mut C;
	// SAFETY: characters must be equal in layout to `u8` bytes
	unsafe { slice::from_raw_parts_mut(ptr, bytes.len()) }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Sequence<C: Character> {
	/// SAFETY: `bytes` must always hold valid `C` characters
	bytes: Bytes,
	marker: PhantomData<C>,
}

impl<C: Character> Deref for Sequence<C> {
	type Target = [C];

	fn deref(&self) -> &[C] {
		let bytes = self.bytes.as_ref();
		// SAFETY: `self.bytes` must always hold valid characters.
		unsafe { b2c(bytes) }
	}
}

impl<C: Character> AsRef<[C]> for Sequence<C> {
	fn as_ref(&self) -> &[C] {
		self
	}
}

impl<C: Character> fmt::Display for Sequence<C> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use fmt::Write;

		for character in self.as_ref() {
			f.write_char(character.to_ascii() as char)?;
		}

		Ok(())
	}
}

impl<C: Character> Sequence<C> {
	pub fn copy_from_slice(data: &[C]) -> Self {
		let bytes = Bytes::copy_from_slice(c2b(data));

		Self {
			bytes,
			marker: PhantomData,
		}
	}

	pub fn from_vec(mut data: Vec<C>) -> Self {
		let ptr = data.as_mut_ptr() as *mut u8;
		let length = data.len();
		let capacity = data.capacity();
		mem::forget(data);

		// SAFETY: characters have the same layout as u8 bytes and the
		// old vector has been forgotten without being dropped.
		let bytes =
			unsafe { Vec::from_raw_parts(ptr, length, capacity) };

		Self {
			bytes: Bytes::from_owner(bytes),
			marker: PhantomData,
		}
	}

	pub fn as_bytes(&self) -> &[u8] {
		&self.bytes
	}

	pub fn len(&self) -> usize {
		self.bytes.len()
	}

	pub fn is_empty(&self) -> bool {
		self.bytes.is_empty()
	}

	pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
		Self {
			bytes: self.bytes.slice(range),
			marker: PhantomData,
		}
	}

	/// Counts how many times the character `c` occurs in the sequence.
	pub fn count(&self, c: C) -> usize {
		let mut out = 0;

		for current in self.as_ref() {
			if *current == c {
				out += 1
			}
		}

		out
	}
}

impl Sequence<DnaNucleotide> {
	pub fn complement(&self) -> Self {
		let mut seq = SequenceMut::from_characters(self);
		seq.complement();
		seq.into()
	}

	pub fn reverse_complement(&self) -> Self {
		let mut seq = SequenceMut::from_characters(self);
		seq.reverse_complement();
		seq.into()
	}
}

#[derive(Debug)]
pub struct SequenceMut<C: Character> {
	bytes: BytesMut,
	marker: PhantomData<C>,
}

impl<C: Character> Deref for SequenceMut<C> {
	type Target = [C];

	fn deref(&self) -> &[C] {
		// SAFETY: `self.bytes` must be valid characters
		unsafe { b2c(self.bytes.as_ref()) }
	}
}

impl<C: Character> DerefMut for SequenceMut<C> {
	fn deref_mut(&mut self) -> &mut [C] {
		// SAFETY: `self.bytes` must be valid characters
		unsafe { b2c_mut(self.bytes.as_mut()) }
	}
}

impl<C: Character> AsRef<[C]> for SequenceMut<C> {
	fn as_ref(&self) -> &[C] {
		self
	}
}

impl<C: Character> AsMut<[C]> for SequenceMut<C> {
	fn as_mut(&mut self) -> &mut [C] {
		// SAFETY: `self.bytes` must be valid characters
		unsafe { b2c_mut(self.bytes.as_mut()) }
	}
}

impl<C: Character> From<SequenceMut<C>> for Sequence<C> {
	fn from(value: SequenceMut<C>) -> Self {
		value.into_sequence()
	}
}

impl<C: Character> SequenceMut<C> {
	pub fn new() -> Self {
		Self {
			bytes: BytesMut::new(),
			marker: PhantomData,
		}
	}

	pub fn with_capacity(capacity: usize) -> Self {
		Self {
			bytes: BytesMut::with_capacity(capacity),
			marker: PhantomData,
		}
	}

	pub fn from_characters(characters: &[C]) -> Self {
		Self {
			bytes: BytesMut::from(c2b(characters)),
			marker: PhantomData,
		}
	}

	pub fn into_sequence(self) -> Sequence<C> {
		Sequence {
			bytes: self.bytes.freeze(),
			marker: PhantomData,
		}
	}

	pub fn reserve(&mut self, additional: usize) {
		self.bytes.reserve(additional)
	}

	pub fn push(&mut self, character: C) {
		self.bytes.put_u8(character.to_byte());
	}

	pub fn extend(&mut self, characters: &[C]) {
		self.bytes.extend_from_slice(c2b(characters));
	}

	pub fn reverse(&mut self) {
		self.bytes.reverse()
	}
}

impl SequenceMut<DnaNucleotide> {
	pub fn complement(&mut self) {
		for base in self.as_mut() {
			*base = base.complement();
		}
	}

	pub fn reverse_complement(&mut self) {
		self.complement();
		self.reverse();
	}
}

impl<C: Character> Default for SequenceMut<C> {
	fn default() -> Self {
		Self {
			bytes: BytesMut::default(),
			marker: PhantomData,
		}
	}
}

#[macro_export]
#[doc(hidden)]
macro_rules! dna {
	($seq:literal) => {
		parse_str::<$crate::DnaNucleotide>($seq)
			.expect("Invalid DNA sequence literal")
			.into_sequence()
	};
}

/// Create a new DNA sequence from a string literal
#[doc(inline)]
pub use crate::dna;
