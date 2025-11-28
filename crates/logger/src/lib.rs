mod macros;
#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::pymodule;

use parking_lot::Mutex;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::Serialize;

use std::{
	fmt::{self, Debug},
	fs::OpenOptions,
	io::{BufWriter, Result as IoResult, Write, sink},
	path::Path,
	sync::OnceLock,
};

pub use serde;

pub static LOGGER: OnceLock<Logger> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "python", pyclass(module = "aspartik.logger"))]
pub enum Level {
	Trace,
	Debug,
	Info,
	Warn,
	Error,
}

pub struct Logger {
	sink: Mutex<BufWriter<Box<dyn Write + Send + Sync>>>,
	targets: Vec<String>,
	min_level: Level,
}

impl Debug for Logger {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("Logger")
	}
}

fn sink_to_field<T>(sink: T) -> Mutex<BufWriter<Box<dyn Write + Send + Sync>>>
where
	T: Write + Send + Sync + 'static,
{
	const CAPACITY: usize = 1024 * 1024; // 1MB
	Mutex::new(BufWriter::with_capacity(CAPACITY, Box::new(sink)))
}

impl Default for Logger {
	fn default() -> Self {
		let sink = sink_to_field(sink());

		Self {
			min_level: Level::Error,
			targets: vec![],
			sink,
		}
	}
}

impl Logger {
	pub fn to_file<P: AsRef<Path>>(mut self, path: P) -> IoResult<Self> {
		let sink = OpenOptions::new()
			.create(true)
			.append(true)
			.open(path)?;
		self.sink = sink_to_field(sink);

		Ok(self)
	}

	pub fn with_level(mut self, level: Level) -> Self {
		self.min_level = level;
		self
	}

	pub fn with_targets(mut self, mut targets: Vec<String>) -> Self {
		self.targets.append(&mut targets);
		self
	}

	pub fn init(self) {
		// SAFETY: TODO
		unsafe { libc::atexit(cleanup) };

		LOGGER.set(self).unwrap();
	}

	fn enabled(&self, level: Level, target: &'static str) -> bool {
		if level < self.min_level {
			return false;
		}

		if self.targets.is_empty() {
			true
		} else {
			self.targets.iter().any(|t| target.starts_with(t))
		}
	}

	pub fn log<T>(&self, kv: &T)
	where
		T: Kv,
	{
		if !self.enabled(kv.level(), kv.target()) {
			return;
		}

		let mut sink = self.sink.lock();
		serde_json::to_writer(&mut *sink, &kv).unwrap();
		sink.write_all(b"\n").unwrap();
	}

	pub fn flush(&self) {
		self.sink.lock().flush().unwrap();
	}
}

extern "C" fn cleanup() {
	let Some(logger) = LOGGER.get() else {
		return;
	};

	logger.flush();
}

#[doc(hidden)]
pub fn one<T>(_: &T) -> usize {
	1
}

#[doc(hidden)]
pub trait Kv: Serialize {
	fn target(&self) -> &'static str;
	fn level(&self) -> Level;
}
