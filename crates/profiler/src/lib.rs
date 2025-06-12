use parking_lot::{Mutex, MutexGuard};

use std::{
	env,
	fs::File,
	io::BufWriter,
	sync::LazyLock,
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub static PROFILER: LazyLock<Profiler> = LazyLock::new(Profiler::new);

#[derive(Debug)]
pub struct Profiler {
	file: Option<Mutex<BufWriter<File>>>,
	start: Option<Mutex<Instant>>,
}

const INACTIVE: Profiler = Profiler {
	file: None,
	start: None,
};

pub const ENV_VAR_NAME: &str = "ASPARTIK_PROFILE";

impl Profiler {
	fn new() -> Profiler {
		if env::var_os(ENV_VAR_NAME).is_none() {
			return INACTIVE;
		}

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

		Profiler {
			file: Some(Mutex::new(BufWriter::new(file))),
			start: Some(Mutex::new(Instant::now())),
		}
	}

	pub fn enabled(&self) -> bool {
		self.file.is_some()
	}

	pub fn start(&self) {
		let start = self.start.as_ref().unwrap();
		*start.lock() = Instant::now();
	}

	pub fn stop(&self) -> Duration {
		let start = self.start.as_ref().unwrap();
		Instant::now() - *start.lock()
	}

	pub fn writer(&self) -> MutexGuard<BufWriter<File>> {
		let file = self.file.as_ref().unwrap();
		file.lock()
	}
}

#[macro_export]
macro_rules! profile {
	(
		target: $target:literal $(,)?
		$($key:ident = $value:expr),*;
		$action:expr
	) => {{
		if !::profiler::PROFILER.enabled() {
			$action
		} else {
			use std::io::Write;

			::profiler::PROFILER.start();
			let out = $action;
			let duration = ::profiler::PROFILER.stop();

			writeln!(
				&mut ::profiler::PROFILER.writer(),
				concat!(
					"{{",
					r#""target":{:?},"#,
					r#""duration":{}"#,
					$(
						",\"",
						stringify!($key),
						"\":",
						r#""{:?}""#,
					)*
					"}}",
				),
				$target,
				duration.as_nanos(),
				$($value),*
			)?;

			out
		}
	}};
}
