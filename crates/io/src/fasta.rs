use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyType};

use crate::{
	reader::StrReader,
	rw::{BoxReader, from_file},
};
use data::{
	DnaNucleotide,
	fasta::{FastaParser, python::PyFastaDnaRecord},
};

#[pyclass(name = "FastaReader", module = "aspartik.io", frozen)]
pub struct PyFastaDnaReader {
	inner: Mutex<StrReader<FastaParser<DnaNucleotide>, BoxReader>>,
}

#[pymethods]
impl PyFastaDnaReader {
	#[classmethod]
	fn from_file(_cls: Py<PyType>, path: &str) -> Result<Self> {
		let parser = FastaParser::new();
		let reader = from_file(path)?;
		Ok(Self {
			inner: StrReader::new(parser, reader).into(),
		})
	}

	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__(&self) -> Option<Result<PyFastaDnaRecord>> {
		let inner = &mut *self.inner.lock();
		inner.next().map(|r| r.map(PyFastaDnaRecord))
	}
}
