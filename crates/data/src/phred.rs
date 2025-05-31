use anyhow::{anyhow, ensure, Context, Error, Result};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The Phred quality score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
	feature = "python",
	pyclass(name = "Phred", module = "aspartik.data", frozen, eq, ord)
)]
pub struct Phred(u8);

impl Phred {
	/// Creates a new Phred quality score from an ASCII character
	///
	/// This function uses the Sanger FASTQ format,
	pub fn new(ch: u8) -> Result<Phred> {
		ensure!(
			(b'!'..=b'I').contains(&ch),
			"Phred quality score must be an ASCII character between '!' (0x21) and 'I' (0x49).  Got {ch:x} instead"
		);
		Ok(Phred(ch - 0x21))
	}

	pub fn accuracy(&self) -> f64 {
		1.0 - self.probability_incorrect()
	}

	/// The chance of an incorrect base call
	pub fn probability_incorrect(&self) -> f64 {
		10.0f64.powf(-f64::from(self.0) / 10.0)
	}
}

impl TryFrom<char> for Phred {
	type Error = Error;

	fn try_from(value: char) -> Result<Self> {
		let byte: u8 = value.try_into().with_context(|| {
			anyhow!("Phred score must be an ASCII character")
		})?;

		Phred::new(byte)
	}
}

impl From<Phred> for char {
	fn from(value: Phred) -> Self {
		(value.0 + 0x21).into()
	}
}

#[cfg(feature = "python")]
#[pymethods]
impl Phred {
	#[new]
	fn py_new(ch: char) -> Result<Self> {
		ch.try_into()
	}

	#[pyo3(name = "accuracy")]
	fn py_accuracy(&self) -> f64 {
		self.accuracy()
	}

	#[pyo3(name = "probability_incorrect")]
	fn py_probability_incorrect(&self) -> f64 {
		self.probability_incorrect()
	}

	fn __repr__(&self) -> String {
		let ch = char::from(*self);
		format!("Phred('{ch}')")
	}
}
