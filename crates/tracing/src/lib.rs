use anyhow::Result;
use parking_lot::Mutex;
use serde_json::{json, to_writer};

use std::{
	env,
	fs::File,
	io::BufWriter,
	sync::LazyLock,
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub static SUBSCRIBER: LazyLock<Tracer> = LazyLock::new(Tracer::new);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
	Error = 0,
	Warn = 1,
	Info = 2,
	Debug = 3,
	Trace = 4,
}

impl Level {
	fn as_str(&self) -> &'static str {
		match self {
			Level::Error => "error",
			Level::Warn => "warn",
			Level::Info => "info",
			Level::Debug => "debug",
			Level::Trace => "trace",
		}
	}
}

#[derive(Debug)]
pub struct Tracer {
	level: Level,
	file: Option<Mutex<BufWriter<File>>>,
	start: Option<Instant>,
}

const INACTIVE: Tracer = Tracer {
	level: Level::Error,
	file: None,
	start: None,
};

pub const LOG_VAR_NAME: &str = "ASPARTIK_LOG";

impl Tracer {
	fn is_active(&self) -> bool {
		self.file.is_some()
	}

	pub fn enabled(&self, level: Level) -> bool {
		self.is_active() && self.level >= level
	}

	fn new() -> Tracer {
		let var = match env::var(LOG_VAR_NAME) {
			Ok(var) => var,
			Err(env::VarError::NotUnicode(_)) => {
				eprintln!("Ignoring LOG_VAR_NAME because it's not a valid Unicode string");
				return INACTIVE;
			}
			Err(env::VarError::NotPresent) => {
				return INACTIVE;
			}
		};

		let level = match var.as_str() {
			"error" => Level::Error,
			"warn" => Level::Warn,
			"info" => Level::Info,
			"debug" => Level::Debug,
			"trace" => Level::Trace,
			_ => {
				eprintln!("Ignoring LOG_VAR_NAME because it's not one of 'error', 'warn', 'info', 'debug', 'trace'");
				return INACTIVE;
			}
		};

		// time since unix epoch in seconds, should be unique enough
		let time = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap()
			.as_secs();

		// XXX: if I pull in a time library, this should print the
		// datetime instead
		let name = format!("aspartik-{time}.log");

		let file = File::create_new(&name).unwrap_or_else(|e| {
			panic!("failed to open a log file at {name}: {e}")
		});

		Tracer {
			level,
			file: Some(Mutex::new(BufWriter::new(file))),
			start: Some(Instant::now()),
		}
	}

	fn elapsed(&self) -> Duration {
		Instant::now().duration_since(self.start.unwrap())
	}

	pub fn event(
		&self,
		level: Level,
		location: &'static str,
	) -> Result<()> {
		if !self.enabled(level) {
			return Ok(());
		}

		let line = json!({
			"type": "event",
			"time": self.elapsed().as_nanos(),
			"level": self.level.as_str(),
			"location": location,
		});

		let file = &self.file.as_ref().unwrap();
		let writer = &mut *file.lock();
		to_writer(writer, &line)?;

		Ok(())
	}
}
