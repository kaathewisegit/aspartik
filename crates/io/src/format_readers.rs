use anyhow::Result;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::{fs::File, path::Path};

use crate::reader::StrReader;
use data::{DnaNucleotide, Msa, PyMsa, fasta::FastaParser};

pub fn read_msa_from_fasta<P: AsRef<Path>>(
	path: P,
) -> Result<Msa<DnaNucleotide>> {
	let io = File::open(path)?;
	let parser = FastaParser::<DnaNucleotide>::new();
	let reader = StrReader::new(parser, io);

	Msa::from_fasta(reader)
}

#[cfg(feature = "python")]
#[pyfunction(name = "read_msa_from_fasta")]
pub fn py_read_msa_from_fasta(path: &str) -> Result<PyMsa> {
	read_msa_from_fasta(path).map(PyMsa)
}
