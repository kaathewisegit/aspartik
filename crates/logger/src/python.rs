use anyhow::Result;
use pyo3::prelude::*;

use std::path::PathBuf;

use crate::{Level, Logger};

/// A builder for the Rust-native logger
///
/// It must be populated with it's methods and initialized with `init`.
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

	/// Filters the events by their target
	///
	/// A logger internally has a list of targets.  If it's empty, all
	/// events are logged.  If it's not, an event's target must have a
	/// prefix from the logger target list, otherwise it won't be logged.
	///
	/// This call is additive.  That is, calling `with_targets` twice will
	/// combine targets from both calls.
	fn with_targets(
		this: Bound<Self>,
		mut targets: Vec<String>,
	) -> Bound<Self> {
		this.borrow_mut().targets.append(&mut targets);
		this
	}

	/// Sets the file path to which the logs will be written
	///
	/// The default path is `b3.log`, which will be created in the CWD.
	fn to_file(this: Bound<Self>, path: PathBuf) -> Bound<Self> {
		this.borrow_mut().path = Some(path);
		this
	}

	/// Sets the logging level
	///
	/// `Error` by default.
	fn with_level(this: Bound<Self>, min_level: Level) -> Bound<Self> {
		this.borrow_mut().min_level = min_level;
		this
	}

	/// Starts the logger
	///
	/// The options can be configured using [`with_targets`][wt],
	/// [`with_level`][wl], and [`to_file`][tf].
	///
	/// [wt]: #Logger.with_targets
	/// [wl]: #Logger.with_level
	/// [tf]: #Logger.to_file
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
