use anyhow::Result;
use parking_lot::Mutex;
use picoarrow::{
	Schema,
	array::{Array, ArrayF64, ArrayU64, NonNullable},
	ipc::{Compression, FileWriter},
};
use pyo3::prelude::*;

use std::{fs::OpenOptions, io::BufWriter};

use crate::mcmc::Mcmc;

const RESULT_FIELDS: [&str; 4] =
	["aborts", "accepts", "prior_rejects", "rejects"];

#[pyclass(module = "aspartik.b3.callbacks", frozen)]
pub struct OperatorStats {
	tunings: Mutex<Vec<ArrayF64<NonNullable>>>,
	results: Mutex<Vec<[ArrayU64<NonNullable>; 4]>>,
	step: Mutex<ArrayU64<NonNullable>>,

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
			results: Mutex::new(Vec::new()),
			step: Mutex::new(ArrayU64::new()),
			path,
			every,
		}
	}

	fn call(&self, py: Python, mcmc: &Mcmc) -> Result<()> {
		self.step.lock().push(mcmc.current_step() as u64);

		let mut tunings = self.tunings.lock();
		let mut results = self.results.lock();

		if tunings.is_empty() {
			for _ in mcmc.scheduler().operators() {
				tunings.push(ArrayF64::new());
				results.push([
					ArrayU64::new(),
					ArrayU64::new(),
					ArrayU64::new(),
					ArrayU64::new(),
				]);
			}
		}

		let stats = mcmc.scheduler().statistics();

		for (i, (tuning, operator)) in tunings
			.iter_mut()
			.zip(mcmc.scheduler().operators())
			.enumerate()
		{
			tuning.push(operator.get_tuning(py)?.unwrap_or(0.0));

			let result_arrays = &mut results[i];
			let s = &stats[i];
			result_arrays[0].push(s.aborts);
			result_arrays[1].push(s.accepts);
			result_arrays[2].push(s.prior_rejects);
			result_arrays[3].push(s.rejects);
		}

		Ok(())
	}

	fn finish(&self, _mcmc: Py<Mcmc>) -> Result<()> {
		let tunings = self.tunings.lock();
		let results = self.results.lock();
		let step = self.step.lock();

		let file = OpenOptions::new()
			.write(true)
			.create(true)
			.truncate(true)
			.open(&self.path)?;

		let writer = BufWriter::new(file);

		let num_operators = tunings.len();
		let mut fields = Vec::with_capacity(num_operators * 5 + 1);

		fields.push(step.make_field("step"));

		for i in 0..num_operators {
			for (j, name) in RESULT_FIELDS.iter().enumerate() {
				fields.push(results[i][j]
					.make_field(&format!("{i}:{name}")));
			}
			fields.push(
				tunings[i].make_field(&format!("{i}:tuning"))
			);
		}

		let schema = Schema::from_fields(fields);
		let mut writer =
			FileWriter::new(writer, schema, Compression::None)?;

		let mut batch_arrays: Vec<&dyn Array> =
			Vec::with_capacity(num_operators * 5 + 1);

		batch_arrays.push(&*step as &dyn Array);

		for i in 0..num_operators {
			for j in 0..4 {
				batch_arrays.push(&results[i][j] as &dyn Array);
			}
			batch_arrays.push(&tunings[i] as &dyn Array);
		}

		writer.write_batch(batch_arrays)?;
		writer.finish()?;

		Ok(())
	}
}
