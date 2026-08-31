use rand::{Rng, RngExt};

use std::fmt::{self, Debug, Write as _};

use crate::nucleotides::DnaNucleotide;

pub mod distance;
mod parse;
#[cfg(feature = "python")]
pub mod python;
mod write;

pub use parse::{parse_append_bytes, parse_append_str, parse_bytes, parse_str};
pub use write::{write_fmt, write_str};

/// A character in a sequence alphabet.
///
/// # Safety
///
/// The type must have the same size an alignment as `u8`, so that `[T]` can be
/// casted to `[u8]`.  In practice this means that the size of the type must be
/// one byte and there are no alignment requirements (all types are 1-byte
/// aligned).
pub unsafe trait Character: Copy + Eq + Debug {
	fn from_ascii(char: u8) -> Option<Self>;

	fn to_ascii(&self) -> u8;

	fn from_byte(b: u8) -> Option<Self>;

	fn into_byte(self) -> u8;
}

#[derive(Debug, Clone, Copy)]
pub struct DisplaySequence<'a, C>(pub &'a [C]);

impl<'a, C: Character> fmt::Display for DisplaySequence<'a, C> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for character in self.0 {
			f.write_char(character.to_ascii() as char)?;
		}
		Ok(())
	}
}

pub fn random_dna<R: Rng>(len: usize, rng: &mut R) -> Vec<DnaNucleotide> {
	let mut seq = Vec::with_capacity(len);
	for __ in 0..len {
		seq.push(match rng.random_range(0..4) {
			0 => DnaNucleotide::Adenine,
			1 => DnaNucleotide::Cytosine,
			2 => DnaNucleotide::Guanine,
			3 => DnaNucleotide::Thymine,
			_ => unreachable!(),
		})
	}
	seq
}

pub fn count<C: Character>(seq: &[C], c: C) -> usize {
	seq.iter().filter(|sc| **sc == c).count()
}

pub fn complement(bases: &mut [DnaNucleotide]) {
	for base in bases {
		*base = base.complement();
	}
}

pub fn reverse_complement(bases: &mut [DnaNucleotide]) {
	complement(bases);
	bases.reverse();
}

#[macro_export]
#[doc(hidden)]
macro_rules! dna {
	($seq:literal) => {
		parse_str::<$crate::DnaNucleotide>($seq)
			.expect("Invalid DNA sequence literal")
	};
}

/// Create a new DNA sequence from a string literal
#[doc(inline)]
pub use crate::dna;
