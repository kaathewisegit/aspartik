use anyhow::{Error, Result, bail};
#[cfg(feature = "arbitrary")]
use arbitrary::Arbitrary;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::fmt;

use crate::seq::Character;

/// The 20 standard amino acids
///
/// This class represents the canonical proteinogenic amino acids using their
/// one-letter [IUPAC codes][ic].  It only supports the 20 standard amino
/// acids, so it does not handle ambiguity codes (`B`, `Z`, `J`, `X`), the
/// stop codon (`*`), selenocysteine (`U`), pyrrolysine (`O`), or gaps (`-`).
///
/// [ic]: https://genome.ucsc.edu/goldenPath/help/iupac.html
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
#[cfg_attr(
	feature = "python",
	pyclass(
		skip_from_py_object,
		name = "AminoAcid",
		module = "aspartik.data",
		frozen,
		eq,
		str,
	)
)]
pub enum AminoAcid {
	Alanine = 0,
	Cysteine = 1,
	AsparticAcid = 2,
	GlutamicAcid = 3,
	Phenylalanine = 4,
	Glycine = 5,
	Histidine = 6,
	Isoleucine = 7,
	Lysine = 8,
	Leucine = 9,
	Methionine = 10,
	Asparagine = 11,
	Proline = 12,
	Glutamine = 13,
	Arginine = 14,
	Serine = 15,
	Threonine = 16,
	Valine = 17,
	Tryptophan = 18,
	Tyrosine = 19,
}

impl fmt::Display for AminoAcid {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use AminoAcid::*;
		let name = match self {
			Alanine => "Alanine",
			Cysteine => "Cysteine",
			AsparticAcid => "Aspartic acid",
			GlutamicAcid => "Glutamic acid",
			Phenylalanine => "Phenylalanine",
			Glycine => "Glycine",
			Histidine => "Histidine",
			Isoleucine => "Isoleucine",
			Lysine => "Lysine",
			Leucine => "Leucine",
			Methionine => "Methionine",
			Asparagine => "Asparagine",
			Proline => "Proline",
			Glutamine => "Glutamine",
			Arginine => "Arginine",
			Serine => "Serine",
			Threonine => "Threonine",
			Valine => "Valine",
			Tryptophan => "Tryptophan",
			Tyrosine => "Tyrosine",
		};
		f.write_str(name)
	}
}

impl AminoAcid {
	fn ident(&self) -> &'static str {
		use AminoAcid::*;
		match self {
			Alanine => "Alanine",
			Cysteine => "Cysteine",
			AsparticAcid => "AsparticAcid",
			GlutamicAcid => "GlutamicAcid",
			Phenylalanine => "Phenylalanine",
			Glycine => "Glycine",
			Histidine => "Histidine",
			Isoleucine => "Isoleucine",
			Lysine => "Lysine",
			Leucine => "Leucine",
			Methionine => "Methionine",
			Asparagine => "Asparagine",
			Proline => "Proline",
			Glutamine => "Glutamine",
			Arginine => "Arginine",
			Serine => "Serine",
			Threonine => "Threonine",
			Valine => "Valine",
			Tryptophan => "Tryptophan",
			Tyrosine => "Tyrosine",
		}
	}
}

// SAFETY: AminoAcid is `repr(u8)`.
unsafe impl Character for AminoAcid {
	fn from_ascii(char: u8) -> Option<Self> {
		use AminoAcid::*;

		Some(match char {
			b'A' | b'a' => Alanine,
			b'C' | b'c' => Cysteine,
			b'D' | b'd' => AsparticAcid,
			b'E' | b'e' => GlutamicAcid,
			b'F' | b'f' => Phenylalanine,
			b'G' | b'g' => Glycine,
			b'H' | b'h' => Histidine,
			b'I' | b'i' => Isoleucine,
			b'K' | b'k' => Lysine,
			b'L' | b'l' => Leucine,
			b'M' | b'm' => Methionine,
			b'N' | b'n' => Asparagine,
			b'P' | b'p' => Proline,
			b'Q' | b'q' => Glutamine,
			b'R' | b'r' => Arginine,
			b'S' | b's' => Serine,
			b'T' | b't' => Threonine,
			b'V' | b'v' => Valine,
			b'W' | b'w' => Tryptophan,
			b'Y' | b'y' => Tyrosine,

			_ => return None,
		})
	}

	fn to_ascii(&self) -> u8 {
		use AminoAcid::*;

		match self {
			Alanine => b'A',
			Cysteine => b'C',
			AsparticAcid => b'D',
			GlutamicAcid => b'E',
			Phenylalanine => b'F',
			Glycine => b'G',
			Histidine => b'H',
			Isoleucine => b'I',
			Lysine => b'K',
			Leucine => b'L',
			Methionine => b'M',
			Asparagine => b'N',
			Proline => b'P',
			Glutamine => b'Q',
			Arginine => b'R',
			Serine => b'S',
			Threonine => b'T',
			Valine => b'V',
			Tryptophan => b'W',
			Tyrosine => b'Y',
		}
	}

	fn from_byte(b: u8) -> Option<Self> {
		use AminoAcid::*;

		Some(match b {
			0 => Alanine,
			1 => Cysteine,
			2 => AsparticAcid,
			3 => GlutamicAcid,
			4 => Phenylalanine,
			5 => Glycine,
			6 => Histidine,
			7 => Isoleucine,
			8 => Lysine,
			9 => Leucine,
			10 => Methionine,
			11 => Asparagine,
			12 => Proline,
			13 => Glutamine,
			14 => Arginine,
			15 => Serine,
			16 => Threonine,
			17 => Valine,
			18 => Tryptophan,
			19 => Tyrosine,

			_ => return None,
		})
	}

	fn into_byte(self) -> u8 {
		self as u8
	}
}

impl TryFrom<char> for AminoAcid {
	type Error = Error;

	fn try_from(value: char) -> Result<Self> {
		let Ok(byte) = value.try_into() else {
			let hex: u32 = value.into();
			bail!(
				"An IUPAC amino acid character must be ASCII, got '{value}' ({hex:x})"
			);
		};

		let Some(amino_acid) = AminoAcid::from_ascii(byte) else {
			bail!(
				"'{value}' is not a valid IUPAC amino acid character"
			)
		};

		Ok(amino_acid)
	}
}

impl From<AminoAcid> for char {
	fn from(value: AminoAcid) -> char {
		value.to_ascii() as char
	}
}

#[cfg(feature = "python")]
#[pymethods]
impl AminoAcid {
	#[new]
	fn new(ch: char) -> Result<Self> {
		Self::try_from(ch)
	}

	fn __repr__(&self) -> String {
		format!("AminoAcid.{}", self.ident())
	}
}
