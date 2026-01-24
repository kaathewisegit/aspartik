use anyhow::{Error, Result, bail};
#[cfg(feature = "arbitrary")]
use arbitrary::Arbitrary;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::fmt;

use crate::seq::Character;

/// DNA nucleotide bases and their combinations
///
/// This class only supports DNA, so it has thymine, but not uracil.  Aside from
/// the core 4 bases it also supports all of their permutations (e.g. `Weak` for
/// adenine or thymine).  See [IUPAC codes][ic] for the full list.  Finally, it
/// also has two special values: `Any`, which encodes an arbitrary base, and
/// `Gap`, which is used in alignments.
///
///
/// ## Protocols
///
/// `DnaNucleotide` implements the `__contains__` protocol for testing which
/// combinations of nucleotides are supersets of others.  For example:
///
/// ```python
/// >>> DNANucleotide.Adenine in DNANucleotide.Weak
/// True
/// >>> DNANucleotide.Pyrimidine in DNANucleotide.NotGuanine
/// True
/// >>> DNANucleotide.Cytosine in DNANucleotide.NotCytosine
/// False
/// ```
//their /
/// `Any` includes all other states, including `Gap`.
///
/// [ic]: https://genome.ucsc.edu/goldenPath/help/iupac.html
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
#[cfg_attr(feature = "arbitrary", derive(Arbitrary))]
#[cfg_attr(
	feature = "python",
	pyclass(
		name = "DNANucleotide",
		module = "aspartik.data",
		frozen,
		eq,
		str,
	)
)]
pub enum DnaNucleotide {
	Adenine = 0b0001,
	Cytosine = 0b0010,
	Guanine = 0b0100,
	Thymine = 0b1000,

	Weak = 0b1001,
	Strong = 0b0110,
	Amino = 0b0011,
	Ketone = 0b1100,
	Purine = 0b0101,
	Pyrimidine = 0b1010,

	NotAdenine = 0b1110,
	NotCytosine = 0b1101,
	NotGuanine = 0b1011,
	NotThymine = 0b0111,

	Any = 0b1111,

	Gap = 0b0000,
}

impl fmt::Display for DnaNucleotide {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use DnaNucleotide::*;
		let name = match self {
			Adenine => "Adenine",
			Cytosine => "Cytosine",
			Guanine => "Guanine",
			Thymine => "Thymine",

			Weak => "Weak",
			Strong => "Strong",
			Amino => "Amino",
			Ketone => "Ketone",
			Purine => "Purine",
			Pyrimidine => "Pyrimidine",

			NotAdenine => "Not adenine",
			NotCytosine => "Not cytosine",
			NotGuanine => "Not guanine",
			NotThymine => "Not thymine",

			Any => "Any",

			Gap => "Gap",
		};
		f.write_str(name)
	}
}

impl DnaNucleotide {
	fn as_u8(&self) -> u8 {
		*self as u8
	}

	pub fn complement(&self) -> Self {
		use DnaNucleotide::*;
		match self {
			Adenine => Thymine,
			Cytosine => Guanine,
			Guanine => Cytosine,
			Thymine => Adenine,

			Weak => Strong,
			Strong => Weak,
			Amino => Ketone,
			Ketone => Amino,
			Purine => Pyrimidine,
			Pyrimidine => Purine,

			NotAdenine => NotThymine,
			NotCytosine => NotGuanine,
			NotGuanine => NotCytosine,
			NotThymine => NotAdenine,

			Any => Any,
			Gap => Gap,
		}
	}

	// XXX: should all states include `Gap`?
	pub fn includes(&self, other: &Self) -> bool {
		(self.as_u8() & other.as_u8()) == other.as_u8()
	}

	/// Probabilistic distribution of a base
	///
	/// Mixed bases such as `Weak` or `NotGuanine` are treated as if all of
	/// their components have the same likelihood of occurring.  `Gap` is
	/// treated same as `Any`.
	pub fn base_frequencies(&self) -> [f64; 4] {
		const F1_3: f64 = 1.0 / 3.0;

		use DnaNucleotide::*;
		match self {
			Adenine => [1.0, 0.0, 0.0, 0.0],
			Cytosine => [0.0, 1.0, 0.0, 0.0],
			Guanine => [0.0, 0.0, 1.0, 0.0],
			Thymine => [0.0, 0.0, 0.0, 1.0],

			Weak => [0.5, 0.0, 0.0, 0.5],
			Strong => [0.0, 0.5, 0.5, 0.0],
			Amino => [0.5, 0.5, 0.0, 0.0],
			Ketone => [0.0, 0.0, 0.5, 0.5],
			Purine => [0.5, 0.0, 0.5, 0.0],
			Pyrimidine => [0.0, 0.5, 0.0, 0.5],

			NotAdenine => [0.0, F1_3, F1_3, F1_3],
			NotCytosine => [F1_3, 0.0, F1_3, F1_3],
			NotGuanine => [F1_3, F1_3, 0.0, F1_3],
			NotThymine => [F1_3, F1_3, F1_3, 0.0],

			Any | Gap => [0.25, 0.25, 0.25, 0.25],
		}
	}

	pub fn base_frequencies_denormalized(&self) -> [f64; 4] {
		use DnaNucleotide::*;
		match self {
			Adenine => [1.0, 0.0, 0.0, 0.0],
			Cytosine => [0.0, 1.0, 0.0, 0.0],
			Guanine => [0.0, 0.0, 1.0, 0.0],
			Thymine => [0.0, 0.0, 0.0, 1.0],

			Weak => [1.0, 0.0, 0.0, 1.0],
			Strong => [0.0, 1.0, 1.0, 0.0],
			Amino => [1.0, 1.0, 0.0, 0.0],
			Ketone => [0.0, 0.0, 1.0, 1.0],
			Purine => [1.0, 0.0, 1.0, 0.0],
			Pyrimidine => [0.0, 1.0, 0.0, 1.0],

			NotAdenine => [0.0, 1.0, 1.0, 1.0],
			NotCytosine => [1.0, 0.0, 1.0, 1.0],
			NotGuanine => [1.0, 1.0, 0.0, 1.0],
			NotThymine => [1.0, 1.0, 1.0, 0.0],

			Any | Gap => [1.0, 1.0, 1.0, 1.0],
		}
	}
}

// SAFETY: DnaNucleotide is `repr(u8)`.
unsafe impl Character for DnaNucleotide {
	fn from_ascii(char: u8) -> Option<Self> {
		use DnaNucleotide::*;

		Some(match char {
			b'A' | b'a' => Adenine,
			b'C' | b'c' => Cytosine,
			b'G' | b'g' => Guanine,
			b'T' | b't' => Thymine,

			b'W' | b'w' => Weak,
			b'S' | b's' => Strong,
			b'M' | b'm' => Amino,
			b'K' | b'k' => Ketone,
			b'R' | b'r' => Purine,
			b'Y' | b'y' => Pyrimidine,

			b'B' | b'b' => NotAdenine,
			b'D' | b'd' => NotCytosine,
			b'H' | b'h' => NotGuanine,
			b'V' | b'v' => NotThymine,

			b'N' | b'n' => Any,
			b'-' => Gap,

			_ => return None,
		})
	}

	fn to_ascii(&self) -> u8 {
		use DnaNucleotide::*;

		match self {
			Adenine => b'A',
			Cytosine => b'C',
			Guanine => b'G',
			Thymine => b'T',

			Weak => b'W',
			Strong => b'S',
			Amino => b'M',
			Ketone => b'K',
			Purine => b'R',
			Pyrimidine => b'Y',

			NotAdenine => b'B',
			NotCytosine => b'D',
			NotGuanine => b'H',
			NotThymine => b'V',

			Any => b'N',
			Gap => b'-',
		}
	}

	#[inline(never)]
	fn from_byte(b: u8) -> Option<Self> {
		use DnaNucleotide::*;

		Some(match b {
			0b0001 => Adenine,
			0b0010 => Cytosine,
			0b0100 => Guanine,
			0b1000 => Thymine,

			0b1001 => Weak,
			0b0110 => Strong,
			0b0011 => Amino,
			0b1100 => Ketone,
			0b0101 => Purine,
			0b1010 => Pyrimidine,

			0b1110 => NotAdenine,
			0b1101 => NotCytosine,
			0b1011 => NotGuanine,
			0b0111 => NotThymine,

			0b1111 => Any,

			0b0000 => Gap,

			_ => return None,
		})
	}

	fn to_byte(&self) -> u8 {
		*self as u8
	}

	fn into_byte(self) -> u8 {
		self as u8
	}
}

impl TryFrom<char> for DnaNucleotide {
	type Error = Error;

	fn try_from(value: char) -> Result<Self> {
		let Ok(byte) = value.try_into() else {
			let hex: u32 = value.into();
			bail!(
				"An IUPAC DNA character must be ASCII, got '{value}' ({hex:x})"
			);
		};

		let Some(nucleotide) = DnaNucleotide::from_ascii(byte) else {
			bail!("'{value}' is not a valid IUPAC DNA character")
		};

		Ok(nucleotide)
	}
}

impl From<DnaNucleotide> for char {
	fn from(value: DnaNucleotide) -> char {
		value.to_ascii() as char
	}
}

#[cfg(feature = "python")]
#[pymethods]
impl DnaNucleotide {
	#[new]
	fn new(ch: char) -> Result<Self> {
		Self::try_from(ch)
	}

	fn __repr__(&self) -> String {
		format!("DNANucleotide.{self}")
	}

	fn __contains__(&self, other: &Self) -> bool {
		self.includes(other)
	}

	/// The complementary pair of the nucleotide base
	///
	/// For combined states the combination of all possible complements will
	/// be returned.  The complement of `Weak` is `Strong`, for `NotGuanine`
	/// it is `NotCytosine`, and so on.  And `Any` and `Gap` will return
	/// themselves.
	#[pyo3(name = "complement")]
	fn py_complement(&self) -> Self {
		self.complement()
	}
}
