use anyhow::Result;
use pyo3::{prelude::*, types::PyType};

use std::{fs::File, io::BufReader, ops::Deref, path::PathBuf};

use crate::{DnaNucleotide, Msa, seq::python::PyDnaSeq};
use rng::PyRng;

/// DNA multiple sequence alignment
///
/// A set of sequences of the same length along with their names.
#[derive(Debug)]
#[pyclass(name = "MSA", module = "aspartik.data.msa", frozen, eq)]
#[repr(transparent)]
pub struct PyMsa(pub Msa<DnaNucleotide>);

impl PartialEq for PyMsa {
	fn eq(&self, other: &Self) -> bool {
		self.0 == other.0
	}
}

impl Deref for PyMsa {
	type Target = Msa<DnaNucleotide>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

#[pymethods]
impl PyMsa {
	// TODO: constructor from two lists: sequences and names

	/// Constructs an MSA from a list of FASTA records
	#[classmethod]
	fn from_fasta_file(_cls: Py<PyType>, path: PathBuf) -> Result<Self> {
		let reader = BufReader::new(File::open(path)?);
		Ok(Self(Msa::from_fasta_reader(reader)?))
	}

	#[classmethod]
	fn random(
		_cls: Py<PyType>,
		num_sequences: usize,
		num_sites: usize,
		names: Vec<String>,
		rng: Py<PyRng>,
	) -> Result<Self> {
		let msa = Msa::random(
			num_sequences,
			num_sites,
			names.into(),
			&mut rng.get().inner(),
		)?;
		Ok(Self(msa))
	}

	/// Total number of sites, including gaps
	#[getter]
	fn num_sites(&self) -> usize {
		self.0.num_sites()
	}

	/// Number of sequences
	#[getter]
	fn num_sequences(&self) -> usize {
		self.0.num_sequences()
	}

	/// The name of the `index`'th sequence
	fn sequence_name(&self, index: usize) -> &str {
		self.0.sequence_name(index)
	}

	/// A list with all of the sequence names
	fn sequence_names(&self) -> &[String] {
		self.0.sequence_names()
	}

	/// `index`'th sequence
	fn sequence(&self, index: usize) -> PyDnaSeq {
		PyDnaSeq(self.0.sequence_owned(index))
	}

	/// The shares each DNA base takes up in the total alignment
	///
	/// Compound bases such as `NotGuanine` are split equiproportionally
	/// between their components.  Gaps are counted the same way as `Any`.
	/// The components of the resulting tuple should always add up to almost
	/// 1, taking floating point precision limitations into account.
	fn base_frequencies(&self) -> (f64, f64, f64, f64) {
		self.0.base_frequencies().into()
	}
}
