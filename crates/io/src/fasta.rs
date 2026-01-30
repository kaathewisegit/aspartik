use anyhow::Result;
use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyType};

use std::io::{BufRead, BufReader};

use crate::rw::AnyReader;
use data::{
	DnaNucleotide,
	fasta::{FastaParser, python::PyFastaDnaRecord},
};

#[pyclass(name = "FastaReader", module = "aspartik.io", frozen)]
pub struct PyFastaDnaReader {
	parser: Mutex<FastaParser<DnaNucleotide>>,
	reader: Mutex<BufReader<AnyReader>>,
}

macro_rules! bubble {
	($e: expr) => {
		match $e {
			Ok(out) => out,
			Err(e) => return Some(Err(e.into())),
		}
	};
}

#[pymethods]
impl PyFastaDnaReader {
	#[classmethod]
	fn from_file(_cls: Py<PyType>, path: &str) -> Result<Self> {
		let reader = AnyReader::from_file(path)?;
		let buf_reader = BufReader::new(reader);
		Ok(Self {
			parser: Mutex::new(FastaParser::new()),
			reader: Mutex::new(buf_reader),
		})
	}

	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__(&self) -> Option<Result<PyFastaDnaRecord>> {
		let parser = &mut *self.parser.lock();
		let reader = &mut *self.reader.lock();

		while !bubble!(reader.fill_buf()).is_empty() {
			// XXX: string buffer which returns str
			let mut src = str::from_utf8(reader.buffer()).unwrap();
			let old_len = src.len();

			let record = bubble!(parser.parse_record(&mut src));
			let new_len = src.len();
			reader.consume(old_len - new_len);

			if let Some(record) = record {
				return Some(Ok(PyFastaDnaRecord(record)));
			}
		}

		if parser.is_done() {
			None
		} else {
			Some(parser.get_final_record().map(PyFastaDnaRecord))
		}
	}
}
