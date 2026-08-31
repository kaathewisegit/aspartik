use pyo3::prelude::*;

use super::Record;
use crate::{
	DnaNucleotide,
	seq::{DisplaySequence, python::PyDnaSeq},
};

/// A FASTA record with a DNA sequence
///
/// Consists of a `DNASeq` and the record id/description.
#[derive(Debug, PartialEq, Eq)]
#[pyclass(name = "DNARecord", module = "aspartik.data.fasta", frozen, eq)]
pub struct PyFastaDnaRecord(pub Record<DnaNucleotide>);

impl AsRef<Record<DnaNucleotide>> for Py<PyFastaDnaRecord> {
	fn as_ref(&self) -> &Record<DnaNucleotide> {
		&self.get().0
	}
}

#[pymethods]
impl PyFastaDnaRecord {
	#[new]
	fn new(mut description: String, sequence: Py<PyDnaSeq>) -> Self {
		if !description.starts_with('>') {
			description.insert(0, '>');
		}
		let record = Record::new(description, sequence.get().0.clone());
		Self(record)
	}

	/// Inner sequence
	///
	/// XXX: this getter currently performs a deep clone of the sequence.
	#[getter]
	fn sequence(&self) -> PyDnaSeq {
		PyDnaSeq(self.0.seq.clone())
	}

	/// Full description, including the starting `>` character
	#[getter]
	fn raw_description(&self) -> String {
		self.0.raw_description().to_owned()
	}

	/// Description of the record
	///
	/// It does not include the starting '>' character, use
	/// `raw_description` for the full string.
	#[getter]
	fn description(&self) -> String {
		self.0.description().to_owned()
	}

	/// ID of the record
	///
	/// ID is conventionally defined as the subset of the description from
	/// the start to the first space character.  This method will return the
	/// full description if there are no spaces in it or an empty string of
	/// the description starts with a space (or is empty itself).
	#[getter]
	fn id(&self) -> String {
		self.0.id().to_string()
	}

	fn __len__(&self) -> usize {
		self.0.seq.len()
	}

	fn __str__(&self) -> String {
		self.0.to_string()
	}

	fn __repr__(&self) -> String {
		format!(
			r#"DNARecord({:?}, DNASeq("{}"))"#,
			self.0.raw_description(),
			DisplaySequence(self.0.sequence()),
		)
	}
}
