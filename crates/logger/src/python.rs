use anyhow::Result;
use pyo3::prelude::*;

use std::path::PathBuf;

use crate::{Level, Logger};

#[derive(Debug)]
#[pyclass(module = "aspartik.logger", name = "Logger")]
struct PyLogger {
	min_level: Level,
	targets: Vec<String>,
	path: Option<PathBuf>,
}

#[pymethods]
impl PyLogger {
	#[new]
	fn new() -> Self {
		Self {
			min_level: Level::Error,
			targets: Vec::new(),
			path: None,
		}
	}

	fn with_targets(
		this: Bound<Self>,
		mut targets: Vec<String>,
	) -> Bound<Self> {
		this.borrow_mut().targets.append(&mut targets);
		this
	}

	fn to_file(this: Bound<Self>, path: PathBuf) -> Bound<Self> {
		this.borrow_mut().path = Some(path);
		this
	}

	fn with_level(this: Bound<Self>, min_level: Level) -> Bound<Self> {
		this.borrow_mut().min_level = min_level;
		this
	}

	fn init(&self) -> Result<()> {
		let path = match &self.path {
			Some(p) => p.clone(),
			None => PathBuf::from("b3.log"),
		};

		Logger::new()
			.with_level(self.min_level)
			.with_targets(self.targets.clone())
			.to_file(path)?
			.init();

		Ok(())
	}
}

#[pymodule(name = "_logger_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use Level;
	#[pymodule_export]
	use PyLogger;

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
