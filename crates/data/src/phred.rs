use anyhow::{Error, Result, bail};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::{fmt, mem};

use crate::seq::Character;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[expect(dead_code)]
enum RangedU8 {
	V0 = 0,
	V1 = 1,
	V2 = 2,
	V3 = 3,
	V4 = 4,
	V5 = 5,
	V6 = 6,
	V7 = 7,
	V8 = 8,
	V9 = 9,
	V10 = 10,
	V11 = 11,
	V12 = 12,
	V13 = 13,
	V14 = 14,
	V15 = 15,
	V16 = 16,
	V17 = 17,
	V18 = 18,
	V19 = 19,
	V20 = 20,
	V21 = 21,
	V22 = 22,
	V23 = 23,
	V24 = 24,
	V25 = 25,
	V26 = 26,
	V27 = 27,
	V28 = 28,
	V29 = 29,
	V30 = 30,
	V31 = 31,
	V32 = 32,
	V33 = 33,
	V34 = 34,
	V35 = 35,
	V36 = 36,
	V37 = 37,
	V38 = 38,
	V39 = 39,
	V40 = 40,
}

/// The [Phred quality score][wiki]
///
/// A Phred score estimates the probability that a given base has been assigned
/// correctly.  It is primarily used in the FASTQ format.
///
///
/// [wiki]: https://en.wikipedia.org/wiki/Phred_quality_score
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
	feature = "python",
	pyclass(
		from_py_object,
		name = "Phred",
		module = "aspartik.data",
		frozen,
		eq,
		ord
	)
)]
pub struct Phred(RangedU8);

impl Phred {
	/// Creates a new Phred quality score from an ASCII character
	///
	/// This function uses the Sanger FASTQ format,
	pub fn new(ch: char) -> Result<Phred> {
		ch.try_into()
	}

	pub fn accuracy(&self) -> f64 {
		1.0 - self.probability_incorrect()
	}

	/// The chance of an incorrect base call
	pub fn probability_incorrect(&self) -> f64 {
		10.0f64.powf(-f64::from(self.into_byte()) / 10.0)
	}
}

impl TryFrom<char> for Phred {
	type Error = Error;

	fn try_from(value: char) -> Result<Self> {
		let Ok(byte) = value.try_into() else {
			bail!("Phred score must be an ASCII character")
		};

		let Some(phred) = Self::from_ascii(byte) else {
			bail!(
				"Phred quality score must be an ASCII character between '!' (0x21) and 'I' (0x49).  Got {byte:x} instead"
			);
		};

		Ok(phred)
	}
}

impl From<&Phred> for char {
	fn from(value: &Phred) -> Self {
		value.to_ascii().into()
	}
}

impl fmt::Debug for Phred {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let ch = char::from(self);
		write!(f, "Phred({ch})")
	}
}

// SAFETY: Phred is `repr(transparent)` over `RangedU8`, which is `repr(u8)`
unsafe impl Character for Phred {
	fn from_ascii(char: u8) -> Option<Self> {
		match char {
			b'!'..=b'I' => {
				let byte = char - 0x21;
				// SAFETY: char is in 0x21..=0x49, so byte is in
				// 0..=40, a valid `RangedU8`.
				let ranged = unsafe {
					mem::transmute::<u8, RangedU8>(byte)
				};
				Some(Phred(ranged))
			}
			_ => None,
		}
	}

	fn to_ascii(&self) -> u8 {
		self.0 as u8 + 0x21
	}

	fn from_byte(b: u8) -> Option<Self> {
		match b {
			0..=40 => {
				// SAFETY: b is in 0..=40
				let ranged = unsafe {
					mem::transmute::<u8, RangedU8>(b)
				};
				Some(Phred(ranged))
			}
			_ => None,
		}
	}

	fn into_byte(self) -> u8 {
		self.0 as u8
	}
}

#[cfg(feature = "python")]
#[pymethods]
impl Phred {
	#[new]
	fn py_new(ch: char) -> Result<Self> {
		ch.try_into()
	}

	/// The chance that a base has been assigned correctly
	#[pyo3(name = "accuracy")]
	fn py_accuracy(&self) -> f64 {
		self.accuracy()
	}

	/// The chance that there has been an error in a base assignment
	///
	/// Equals `1 - self.accuracy()`.
	#[pyo3(name = "probability_incorrect")]
	fn py_probability_incorrect(&self) -> f64 {
		self.probability_incorrect()
	}

	fn __repr__(&self) -> String {
		let ch = char::from(self);
		format!("Phred('{ch}')")
	}
}
