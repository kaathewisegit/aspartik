use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use picoarrow::{
	Field, Schema,
	array::{
		Array, ArrayBinary, ArrayF64, ArrayFixedSizeList, ArrayU8,
		ArrayU64, NonNullable,
	},
	ipc::{Compression, FileWriter},
};
use pyo3::{prelude::*, types::PyDict};

use std::{
	fs::{File, OpenOptions},
	io::BufWriter,
	path::PathBuf,
};

use crate::{
	mcmc::Mcmc,
	parameters::{Parameter, PyClassVector, PyReal, PyRealVector, PyTree},
	priors::PyPrior,
};

enum LoggedArray {
	Real {
		real: Py<PyReal>,
		array: ArrayF64<NonNullable>,
	},
	RealVector {
		vector: Py<PyRealVector>,
		array: ArrayFixedSizeList<ArrayF64<NonNullable>, NonNullable>,
	},
	ClassVector {
		classvec: Py<PyClassVector>,
		array: ArrayFixedSizeList<ArrayU8<NonNullable>, NonNullable>,
	},
	Tree {
		tree: Py<PyTree>,
		binary: ArrayBinary<NonNullable>,
		length: ArrayF64<NonNullable>,
		height: ArrayF64<NonNullable>,
	},
	Prior {
		prior: PyPrior,
		array: ArrayF64<NonNullable>,
	},
}

struct Item {
	name: String,
	array: LoggedArray,
}

impl Item {
	fn update(&mut self, py: Python) -> Result<()> {
		match &mut self.array {
			LoggedArray::Real { real, array } => {
				array.push(real.get().inner().value());
			}
			LoggedArray::RealVector { vector, array } => {
				let vector = vector.get().inner();
				array.push(|n| {
					for i in 0..vector.len() {
						n.push(vector[i]);
					}
				})?;
			}
			LoggedArray::ClassVector { classvec, array } => {
				let classvec = classvec.get().inner();
				array.push(|n| {
					for i in 0..classvec.len() {
						n.push(classvec[i]);
					}
				})?;
			}
			LoggedArray::Prior { prior, array } => {
				let value = prior.probability(py)?;
				array.push(value);
			}
			LoggedArray::Tree {
				tree,
				binary,
				length,
				height,
			} => {
				let tree = &*tree.get().inner();
				length.push(tree.total_length());
				height.push(tree.height_of(*tree.root()));

				let mut out = Vec::new();
				tree.dump(&mut out)?;
				binary.push(&out)?;
			}
		}

		Ok(())
	}

	fn memory_size(&self) -> usize {
		match &self.array {
			LoggedArray::Real { array, .. } => array.memory_size(),
			LoggedArray::RealVector { array, .. } => {
				array.memory_size()
			}
			LoggedArray::ClassVector { array, .. } => {
				array.memory_size()
			}
			LoggedArray::Prior { array, .. } => array.memory_size(),
			LoggedArray::Tree {
				binary,
				length,
				height,
				..
			} => {
				length.memory_size()
					+ height.memory_size() + binary.memory_size()
			}
		}
	}
}

#[pyclass(module = "aspartik.b3.callbacks", frozen)]
pub struct TraceWriter {
	items: Mutex<Vec<Item>>,

	step: Mutex<ArrayU64<NonNullable>>,
	posterior: Mutex<ArrayF64<NonNullable>>,
	prior: Mutex<ArrayF64<NonNullable>>,
	likelihood: Mutex<ArrayF64<NonNullable>>,

	writer: Mutex<FileWriter<BufWriter<File>>>,

	#[pyo3(get)]
	every: usize,
}

#[pymethods]
impl TraceWriter {
	#[new]
	#[pyo3(signature = (items, path,
		*,
        	zstd = false, overwrite = false, every
	))]
	fn new(
		items: Bound<'_, PyDict>,
		path: PathBuf,
		zstd: bool,
		overwrite: bool,
		every: usize,
	) -> Result<Self> {
		let mut items_v = Vec::new();

		for (key, val) in items {
			let name = key.extract::<String>()?;

			let array = if let Ok(real) = val.cast::<PyReal>() {
				LoggedArray::Real {
					real: real.clone().unbind(),
					array: ArrayF64::new(),
				}
			} else if let Ok(real_vector) =
				val.cast::<PyRealVector>()
			{
				let size =
					real_vector.get().inner().len() as i32;
				LoggedArray::RealVector {
					vector: real_vector.clone().unbind(),
					array: ArrayFixedSizeList::new(
						ArrayF64::new(),
						size,
					),
				}
			} else if let Ok(class_vector) =
				val.cast::<PyClassVector>()
			{
				let size =
					class_vector.get().inner().len() as i32;
				LoggedArray::ClassVector {
					classvec: class_vector.clone().unbind(),
					array: ArrayFixedSizeList::new(
						ArrayU8::new(),
						size,
					),
				}
			} else if let Ok(tree) = val.cast::<PyTree>() {
				LoggedArray::Tree {
					tree: tree.clone().unbind(),
					binary: ArrayBinary::new(),
					length: ArrayF64::new(),
					height: ArrayF64::new(),
				}
			} else if let Ok(prior) = val.extract::<PyPrior>() {
				LoggedArray::Prior {
					prior,
					array: ArrayF64::new(),
				}
			} else {
				unimplemented!();
			};

			items_v.push(Item { name, array });
		}

		let compression = if zstd {
			Compression::Zstd(5)
		} else {
			Compression::None
		};

		if path.is_file() && path.exists() && !overwrite {
			bail!(
				"File {path:?} already exists. Add `overwrite=True` to replace it"
			);
		}

		let file = OpenOptions::new()
			.write(true)
			.create(true)
			.truncate(true)
			.create_new(!overwrite)
			.open(&path)
			.with_context(|| {
				format!(
					"Failed to create the trace file {}",
					path.display()
				)
			})?;
		let writer = BufWriter::new(file);

		let step = ArrayU64::new();
		let posterior = ArrayF64::new();
		let prior = ArrayF64::new();
		let likelihood = ArrayF64::new();

		let mut fields: Vec<Field> = vec![
			step.make_field("step"),
			posterior.make_field("posterior"),
			prior.make_field("prior"),
			likelihood.make_field("likelihood"),
		];
		items_to_fields(&items_v, &mut fields);

		let schema = Schema::from_fields(fields);
		let writer = FileWriter::new(writer, schema, compression)?;

		Ok(Self {
			items: Mutex::new(items_v),
			step: Mutex::new(step),
			posterior: Mutex::new(posterior),
			prior: Mutex::new(prior),
			likelihood: Mutex::new(likelihood),
			writer: Mutex::new(writer),
			every,
		})
	}

	fn call(&self, py: Python, mcmc: &Mcmc) -> Result<()> {
		self.step.lock().push(mcmc.current_step() as u64);
		self.posterior.lock().push(mcmc.posterior());
		self.prior.lock().push(mcmc.prior(py)?);
		self.likelihood.lock().push(mcmc.likelihood_value()?);

		let mut total_mem: usize = 0;

		let mut items = self.items.lock();
		for item in &mut *items {
			item.update(py)?;
			total_mem += item.memory_size();
		}
		drop(items);

		if total_mem >= 100_000_000 {
			self.write_batch()?;
		}

		Ok(())
	}

	fn finish(&self, _mcmc: Py<Mcmc>) -> Result<()> {
		self.write_batch()?;
		self.writer.lock().finish()?;
		Ok(())
	}
}

impl TraceWriter {
	fn write_batch(&self) -> Result<()> {
		let mut step = self.step.lock();
		let mut posterior = self.posterior.lock();
		let mut prior = self.prior.lock();
		let mut likelihood = self.likelihood.lock();

		let mut batch_arrays: Vec<&dyn Array> = vec![
			&*step as &dyn Array,
			&*posterior as &dyn Array,
			&*prior as &dyn Array,
			&*likelihood as &dyn Array,
		];

		let mut items = self.items.lock();
		for item in items.iter() {
			match &item.array {
				LoggedArray::Real { array, .. } => {
					batch_arrays.push(array as &dyn Array)
				}
				LoggedArray::RealVector { array, .. } => {
					batch_arrays.push(array as &dyn Array)
				}
				LoggedArray::ClassVector { array, .. } => {
					batch_arrays.push(array as &dyn Array)
				}
				LoggedArray::Prior { array, .. } => {
					batch_arrays.push(array as &dyn Array)
				}
				LoggedArray::Tree {
					binary,
					length,
					height,
					..
				} => {
					batch_arrays.push(binary as &dyn Array);
					batch_arrays.push(length as &dyn Array);
					batch_arrays.push(height as &dyn Array);
				}
			}
		}

		let mut writer = self.writer.lock();
		writer.write_batch(batch_arrays)?;

		step.clear();
		posterior.clear();
		prior.clear();
		likelihood.clear();

		for item in items.iter_mut() {
			match &mut item.array {
				LoggedArray::Real { array, .. } => {
					array.clear();
				}
				LoggedArray::RealVector { array, .. } => {
					array.clear();
				}
				LoggedArray::ClassVector { array, .. } => {
					array.clear();
				}
				LoggedArray::Prior { array, .. } => {
					array.clear();
				}
				LoggedArray::Tree {
					binary,
					length,
					height,
					..
				} => {
					binary.clear();
					length.clear();
					height.clear();
				}
			}
		}

		Ok(())
	}
}

fn items_to_fields(items: &[Item], fields: &mut Vec<Field>) {
	for item in items {
		match &item.array {
			LoggedArray::Real { array, .. } => {
				fields.push(array.make_field(&item.name))
			}
			LoggedArray::RealVector { array, .. } => {
				fields.push(array.make_field(&item.name))
			}
			LoggedArray::ClassVector { array, .. } => {
				fields.push(array.make_field(&item.name))
			}
			LoggedArray::Prior { array, .. } => {
				fields.push(array.make_field(&item.name))
			}
			LoggedArray::Tree {
				binary,
				length,
				height,
				..
			} => {
				fields.push(binary.make_field(&item.name));
				fields.push(length.make_field(&format!(
					"{}:length",
					item.name
				)));
				fields.push(height.make_field(&format!(
					"{}:height",
					item.name
				)));
			}
		}
	}
}
