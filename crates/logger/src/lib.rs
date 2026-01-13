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

pub fn logger() -> &'static Logger {
	LOGGER.get_or_init(Logger::default)
}

/// Log verbosity level
///
/// It's a reversed version of [`log`'s `Level`][ll].  The least important
/// events have the lower value, so `Trace < Error`.
///
/// [ll]: https://docs.rs/log/latest/log/enum.Level.html
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "python", pyclass(module = "aspartik.logger"))]
pub enum Level {
	/// All logged events
	///
	/// Beware of using this level!  It can generate hundreds of megabytes
	/// of logs in seconds.
	Trace,
	/// Logs important but mundane events, such as object creation
	Debug,
	/// User-facing tips
	///
	/// Can be used for configuraiton notes or optimization suggestions.
	Info,
	/// Important, but not fatal errors
	///
	/// Should include potential (but not panic-y) misconfiguration or
	/// deprecation warnings.  It should be used for things which could be
	/// problematic, but probably aren't.
	Warn,
	/// Severe errors which could compromise the results
	///
	/// Reserved for events which signal errors which might corrupt the
	/// output or the analysis without erroring out the standard way.
	Error,
}

pub struct Logger {
	destination: Mutex<BufWriter<Box<dyn Write + Send + Sync>>>,
	targets: Vec<String>,
	min_level: Level,
}

impl Debug for Logger {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Logger")
			.field("targets", &self.targets)
			.field("min_level", &self.min_level)
			.finish()
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
			destination: sink,
		}
	}
}

impl Logger {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn to_file<P: AsRef<Path>>(mut self, path: P) -> IoResult<Self> {
		let sink = OpenOptions::new()
			.create(true)
			.append(true)
			.open(path)?;
		self.destination = sink_to_field(sink);

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
		// SAFETY: `atexit` is thread safe, so `init` can be called from
		// various threads.  Calling `exit` in the cleanup function is
		// undefined behavior, but `cleanup` never panics.
		//
		// https://www.man7.org/linux/man-pages/man3/atexit.3.html
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

		let mut sink = self.destination.lock();
		serde_json::to_writer(&mut *sink, &kv).unwrap();
		sink.write_all(b"\n").unwrap();
	}

	pub fn flush(&self) {
		self.destination.lock().flush().unwrap();
	}
}

/// Flushes the logger if it hasn't done so yet
///
/// This function never panics.  If flushing fails, the error is simply printed
/// and the data is discarded.
extern "C" fn cleanup() {
	let Some(logger) = LOGGER.get() else {
		return;
	};

	if let Err(error) = logger.destination.lock().flush() {
		eprintln!("Failed to flush the logs while exiting: {error}");
	}
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
