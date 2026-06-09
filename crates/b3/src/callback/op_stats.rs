use anyhow::Result;
use parking_lot::Mutex;
use picoarrow::{
	Field, Schema,
	array::{Array, ArrayF64, NonNullable},
	ipc::{Compression, FileWriter},
};
use pyo3::prelude::*;

use std::{fs::OpenOptions, io::BufWriter};

use crate::mcmc::Mcmc;

#[pyclass(module = "aspartik.b3.callbacks", frozen)]
pub struct OperatorStats {
	tunings: Mutex<Vec<ArrayF64<NonNullable>>>,

	path: String,

	#[pyo3(get)]
	every: usize,
}

#[pymethods]
impl OperatorStats {
	#[new]
	fn new(path: String, every: usize) -> Self {
		Self {
			tunings: Mutex::new(Vec::new()),
			path,
			every,
		}
	}

	fn call(&self, py: Python, mcmc: &Mcmc) -> Result<()> {
		let mut tunings = self.tunings.lock();
		if tunings.is_empty() {
			init(mcmc, &mut tunings);
		}

		for (tuning, operator) in
			tunings.iter_mut().zip(mcmc.scheduler().operators())
		{
			tuning.push(operator.get_tuning(py)?.unwrap_or(0.0));
		}

		Ok(())
	}

	fn finish(&self, _mcmc: Py<Mcmc>) -> Result<()> {
		let tunings = self.tunings.lock();

		let file = OpenOptions::new()
			.write(true)
			.create(true)
			.truncate(true)
			.open(&self.path)?;

		let writer = BufWriter::new(file);

		let fields: Vec<Field> = tunings
			.iter()
			.enumerate()
			.map(|(i, tuning)| {
				tuning.make_field(&format!("{i}:tuning"))
			})
			.collect();

		let schema = Schema::from_fields(fields);
		let mut writer =
			FileWriter::new(writer, schema, Compression::Zstd(5))?;

		let batch_arrays: Vec<&dyn Array> =
			tunings.iter().map(|t| t as &dyn Array).collect();
		writer.write_batch(batch_arrays)?;
		writer.finish()?;

		Ok(())
	}
}

fn init(mcmc: &Mcmc, tunings: &mut Vec<ArrayF64<NonNullable>>) {
	for _ in mcmc.scheduler().operators() {
		tunings.push(ArrayF64::new());
	}
}
