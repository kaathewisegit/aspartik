use anyhow::{Context, Result, bail};
use arrow_array::{
	RecordBatch,
	builder::{
		ArrayBuilder, BinaryBuilder, Float64Builder, ListBuilder,
		UInt8Builder, UInt64Builder,
	},
};
use arrow_ipc::{
	CompressionType,
	writer::{FileWriter, IpcWriteOptions},
};
use arrow_schema::{DataType, Field, SchemaBuilder, SchemaRef};
use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};

use std::{
	collections::HashMap,
	fs::{File, OpenOptions},
	io::BufWriter,
	path::PathBuf,
	sync::Arc,
};

use crate::{
	mcmc::Mcmc,
	parameters::{Parameter, PyClassVector, PyReal, PyRealVector, PyTree},
	priors::PyPrior,
};

type Arrays = HashMap<String, Box<dyn ArrayBuilder>>;

#[pyclass(module = "aspartik.b3.callbacks", frozen)]
pub struct TraceWriter {
	items: Vec<(String, Py<PyAny>)>,
	arrays: Mutex<Arrays>,
	schema: SchemaRef,
	writer: Mutex<FileWriter<BufWriter<File>>>,
	#[pyo3(get)]
	every: usize,
}

#[pymethods]
impl TraceWriter {
	#[new]
	#[pyo3(signature = (
		items, path,
		*,
		zstd = false, overwrite = false, every
	))]
	fn new(
		py: Python,
		items: Bound<'_, PyDict>,
		path: PathBuf,
		zstd: bool,
		overwrite: bool,
		every: usize,
	) -> Result<Self> {
		let items = dict_to_vec(items)?;

		let mut arrays = HashMap::new();
		let mut schema = SchemaBuilder::new();

		schema.push(field("step", DataType::UInt64));
		arrays.insert(
			"step".to_owned(),
			dyn_builder(UInt64Builder::new()),
		);
		for name in ["posterior", "prior", "likelihood"] {
			schema.push(field(name, DataType::Float64));
			arrays.insert(
				name.to_owned(),
				dyn_builder(Float64Builder::new()),
			);
		}

		for (name, value) in &items {
			init_value(
				name,
				value.bind(py),
				&mut schema,
				&mut arrays,
			)?;
		}

		let schema = schema.finish();

		let compression = if zstd {
			Some(CompressionType::ZSTD)
		} else {
			None
		};

		if path.is_file() && path.exists() && !overwrite {
			bail!(
				"File {path:?} already exists.  Add `overwrite=True` to replace it"
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
		let writer = FileWriter::try_new_with_options(
			writer,
			&schema,
			IpcWriteOptions::default()
				.try_with_compression(compression)?,
		)?;
		let writer = Mutex::new(writer);

		Ok(Self {
			items,
			arrays: Mutex::new(arrays),
			schema: schema.into(),
			writer,
			every,
		})
	}

	fn call(&self, py: Python, mcmc: &Mcmc) -> Result<()> {
		let mut arrays_lock = self.arrays.lock();
		let arrays = &mut *arrays_lock;

		let array = get::<UInt64Builder>(arrays, "step");
		array.append_value(mcmc.current_step() as u64);

		let array = get::<Float64Builder>(arrays, "posterior");
		array.append_value(mcmc.posterior());

		let array = get::<Float64Builder>(arrays, "prior");
		array.append_value(mcmc.prior(py)?);

		let array = get::<Float64Builder>(arrays, "likelihood");
		array.append_value(mcmc.likelihood_value()?);

		let mut max_size = 0;
		for (name, value) in &self.items {
			let size = write_value(name, value.bind(py), arrays)?;
			max_size = max_size.max(size);
		}

		drop(arrays_lock);

		// 64MB
		if max_size > 64_000_000 {
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
		let arrays = &mut *self.arrays.lock();
		let mut columns = Vec::new();
		for field in self.schema.fields() {
			columns.push(arrays
				.get_mut(field.name())
				.unwrap()
				.finish());
		}

		let batch = RecordBatch::try_new(self.schema.clone(), columns)?;

		self.writer.lock().write(&batch)?;

		Ok(())
	}
}

fn write_value(
	name: &str,
	value: &Bound<'_, PyAny>,
	arrays: &mut Arrays,
) -> Result<usize> {
	let size: usize;

	if let Ok(real) = value.cast_exact::<PyReal>() {
		let array = get::<Float64Builder>(arrays, name);
		array.append_value(real.get().inner().value());
		size = array.len() * 8;
	} else if let Ok(real_vector) = value.cast_exact::<PyRealVector>() {
		let array = get::<ListBuilder<Float64Builder>>(arrays, name);
		let subarr = array.values();
		for value in real_vector.get().inner().iter() {
			subarr.append_value(*value);
		}
		array.append(true);
		size = *array.offsets_slice().last().unwrap() as usize;
	} else if let Ok(class_vector) = value.cast_exact::<PyClassVector>() {
		let array = get::<ListBuilder<UInt8Builder>>(arrays, name);
		for value in class_vector.get().inner().iter() {
			array.values().append_value(*value);
		}
		array.append(true);
		size = *array.offsets_slice().last().unwrap() as usize;
	} else if let Ok(tree) = value.cast_exact::<PyTree>() {
		let tree = &*tree.get().inner();

		let array = get::<BinaryBuilder>(arrays, name);
		tree.dump(array)?;
		array.append_value("");
		size = *array.offsets_slice().last().unwrap() as usize;

		let name_length = format!("{name}:length");
		let array = get::<Float64Builder>(arrays, &name_length);
		array.append_value(tree.total_length());

		let name_height = format!("{name}:height");
		let array = get::<Float64Builder>(arrays, &name_height);
		array.append_value(tree.height_of(*tree.root()));
	} else if let Ok(prior) = value.extract::<PyPrior>() {
		let array = get::<Float64Builder>(arrays, name);
		array.append_value(prior.probability(value.py())?);
		size = array.len() * 8;
	} else {
		unreachable!();
	}

	Ok(size)
}

fn init_value(
	name: &str,
	value: &Bound<'_, PyAny>,
	schema: &mut SchemaBuilder,
	arrays: &mut Arrays,
) -> Result<()> {
	if value.is_exact_instance_of::<PyReal>() {
		schema.push(field(name, DataType::Float64));
		arrays.insert(
			name.to_owned(),
			dyn_builder(Float64Builder::new()),
		);
	} else if value.is_exact_instance_of::<PyRealVector>() {
		schema.push(list_field(name, DataType::Float64));
		arrays.insert(
			name.to_owned(),
			dyn_builder(ListBuilder::new(Float64Builder::new())),
		);
	} else if value.is_exact_instance_of::<PyClassVector>() {
		schema.push(list_field(name, DataType::UInt8));
		arrays.insert(
			name.to_owned(),
			dyn_builder(ListBuilder::new(UInt8Builder::new())),
		);
	} else if value.is_exact_instance_of::<PyTree>() {
		schema.push(field(name, DataType::Binary));
		arrays.insert(
			name.to_owned(),
			dyn_builder(BinaryBuilder::new()),
		);

		let name_length = format!("{name}:length");
		schema.push(field(&name_length, DataType::Float64));
		arrays.insert(name_length, dyn_builder(Float64Builder::new()));

		let name_height = format!("{name}:height");
		schema.push(field(&name_height, DataType::Float64));
		arrays.insert(name_height, dyn_builder(Float64Builder::new()));
	} else if let Ok(_value) = value.extract::<PyPrior>() {
		schema.push(field(name, DataType::Float64));
		arrays.insert(
			name.to_owned(),
			dyn_builder(Float64Builder::new()),
		);
	} else {
		bail!(
			"Unsupported logging target: {}",
			value.get_type().name()?,
		);
	}

	Ok(())
}

fn dyn_builder<B: ArrayBuilder>(b: B) -> Box<dyn ArrayBuilder> {
	Box::new(b) as Box<dyn ArrayBuilder>
}

/// All fields are marked as nullable.  Now, in practice they will never be
/// null, so it'd be nice to mark them as such.  Unfortunately, trying to do so
/// causes a chain reaction of poorly designed Box<dyn> dominoes to fall through
/// this entire implementation:
///
/// - List builders store field information inside them, so they must also be
///   marked as non-nullable.
///
/// - There are no methods to mark builders as non-nullable, one must use
///   `with_field`.
///
/// - This means that a function creating a list builder must take both a
///   DataType, and a builder for that data type, which is redundant.
///
/// - `make_builder` doesn't help, because it returns `Box<dyn ArrayBuilder>`,
///   and so the return type for a list data type will be
///   `ListBuilder<Box<DtBuilder>>` with added `dyn` indirection.
///
/// - The latter makes working with `Box<dyn ArrayBuilder>` even more painful.
fn field(name: &str, dt: DataType) -> Field {
	Field::new(name, dt, true)
}

fn list_field(name: &str, dt: DataType) -> Field {
	field(name, DataType::List(Arc::new(field("item", dt))))
}

fn get<'a, T: 'static>(
	arrays: &'a mut HashMap<String, Box<dyn ArrayBuilder>>,
	name: &str,
) -> &'a mut T {
	arrays.get_mut(name)
		.unwrap()
		.as_any_mut()
		.downcast_mut::<T>()
		.unwrap()
}

fn dict_to_vec(dict: Bound<'_, PyDict>) -> Result<Vec<(String, Py<PyAny>)>> {
	let mut out = vec![];

	for (key, value) in dict {
		let key = key.extract::<String>()?;

		out.push((key, value.unbind()))
	}

	Ok(out)
}
