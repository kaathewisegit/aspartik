use anyhow::Error;
use thiserror::Error;

use std::fmt;

#[derive(Debug, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum DnaNucleotideError {
	#[error("'{0}' not a valid IUPAC nucleotide code character")]
	InvalidChar(char),
	#[error("{0:X} is not a valid Aspartik nucleotide binary representation")]
	InvalidByte(u8),
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum DnaNucleotide {
	Adenine = 0b0001,
	Cytosine = 0b0010,
	Guanine = 0b0100,
	Thymine = 0b1000,

	Weak = 0b1001,
	Strong = 0b0110,
	Amino = 0b1100,
	Ketone = 0b0011,
	Purine = 0b0101,
	Pyrimidine = 0b1010,

	NotAdenine = 0b1110,
	NotCytosine = 0b1101,
	NotGuanine = 0b1011,
	NotThymine = 0b0111,

	Any = 0b1111,

	Gap = 0b1_0000,
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

impl From<DnaNucleotide> for u8 {
	fn from(value: DnaNucleotide) -> Self {
		value as u8
	}
}

impl TryFrom<u8> for DnaNucleotide {
	type Error = Error;

	fn try_from(value: u8) -> Result<Self, Self::Error> {
		Ok(match value {
			0b0001 => Self::Adenine,
			0b0010 => Self::Cytosine,
			0b0100 => Self::Guanine,
			0b1000 => Self::Thymine,

			0b1001 => Self::Weak,
			0b0110 => Self::Strong,
			0b1100 => Self::Amino,
			0b0011 => Self::Ketone,
			0b0101 => Self::Purine,
			0b1010 => Self::Pyrimidine,

			0b1110 => Self::NotAdenine,
			0b1101 => Self::NotCytosine,
			0b1011 => Self::NotGuanine,
			0b0111 => Self::NotThymine,

			0b1111 => Self::Any,

			0b1_0000 => Self::Gap,

			_ => Err(DnaNucleotideError::InvalidByte(value))?,
		})
	}
}

impl TryFrom<char> for DnaNucleotide {
	type Error = Error;

	// https://genome.ucsc.edu/goldenPath/help/iupac.html
	fn try_from(value: char) -> Result<Self, Self::Error> {
		use DnaNucleotide::*;
		Ok(match value {
			'A' => Adenine,
			'C' => Cytosine,
			'G' => Guanine,
			'T' => Thymine,

			'W' => Weak,
			'S' => Strong,
			'M' => Amino,
			'K' => Ketone,
			'R' => Purine,
			'Y' => Pyrimidine,

			'B' => NotAdenine,
			'D' => NotCytosine,
			'H' => NotGuanine,
			'V' => NotThymine,

			'N' => Any,
			'-' => Gap,

			_ => Err(DnaNucleotideError::InvalidChar(value))?,
		})
	}
}

impl From<DnaNucleotide> for char {
	fn from(value: DnaNucleotide) -> char {
		use DnaNucleotide::*;
		match value {
			Adenine => 'A',
			Cytosine => 'C',
			Guanine => 'G',
			Thymine => 'T',

			Weak => 'W',
			Strong => 'S',
			Amino => 'M',
			Ketone => 'K',
			Purine => 'R',
			Pyrimidine => 'Y',

			NotAdenine => 'B',
			NotCytosine => 'D',
			NotGuanine => 'H',
			NotThymine => 'V',

			Any => 'N',
			Gap => '-',
		}
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

	pub fn includes(&self, other: &Self) -> bool {
		(self.as_u8() & other.as_u8()) == other.as_u8()
	}
}
