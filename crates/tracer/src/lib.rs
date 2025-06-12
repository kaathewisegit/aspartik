use hashbrown::HashMap;
use parking_lot::Mutex;
use serde_json::{json, to_writer};
use tracing_core::{
	dispatcher::{set_global_default, SetGlobalDefaultError},
	span::{Attributes, Id, Record},
	Event, Level, Metadata, Subscriber,
};

use std::{
	env,
	fs::File,
	io::{BufWriter, Write},
	sync::atomic::{AtomicU64, Ordering},
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Copy)]
struct SpanData {
	start: Instant,
	total: Duration,
}

impl SpanData {
	fn new() -> Self {
		SpanData {
			start: Instant::now(),
			total: Duration::default(),
		}
	}
}

#[derive(Debug)]
pub struct Tracer {
	level: Level,
	counter: AtomicU64,
	file: Mutex<BufWriter<File>>,
	spans: Mutex<HashMap<Id, SpanData>>,
}

pub const LOG_VAR_NAME: &str = "ASPARTIK_LOG";

impl Tracer {
	pub fn new() -> Option<Tracer> {
		let var = match env::var(LOG_VAR_NAME) {
			Ok(var) => var,
			Err(env::VarError::NotUnicode(_)) => {
				eprintln!("Ignoring LOG_VAR_NAME because it's not a valid Unicode string");
				return None;
			}
			Err(env::VarError::NotPresent) => {
				return None;
			}
		};

		let level = match var.as_str() {
			"error" => Level::ERROR,
			"warn" => Level::WARN,
			"info" => Level::INFO,
			"debug" => Level::DEBUG,
			"trace" => Level::TRACE,
			_ => {
				eprintln!("Ignoring LOG_VAR_NAME because it's not one of 'error', 'warn', 'info', 'debug', 'trace'");
				return None;
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

		Some(Tracer {
			level,
			// span IDs must be more than 0
			counter: AtomicU64::new(1),
			file: Mutex::new(BufWriter::new(file)),
			spans: Mutex::new(HashMap::new()),
		})
	}

	pub fn init() -> Result<(), SetGlobalDefaultError> {
		if let Some(tracer) = Tracer::new() {
			set_global_default(tracer.into())
		} else {
			Ok(())
		}
	}
}

impl Subscriber for Tracer {
	fn enabled(&self, metadata: &Metadata<'_>) -> bool {
		*metadata.level() >= self.level
	}

	fn new_span(&self, _span: &Attributes<'_>) -> Id {
		let value = self.counter.fetch_add(1, Ordering::Relaxed);
		let id = Id::from_u64(value);

		let spans = &mut *self.spans.lock();
		spans.insert(id.clone(), SpanData::new());

		id
	}

	fn record(&self, _span: &Id, _values: &Record<'_>) {}

	fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

	// TODO
	fn event(&self, _event: &Event<'_>) {}

	fn enter(&self, span: &Id) {
		let spans = &mut *self.spans.lock();
		spans.entry(span.clone())
			.and_modify(|data| data.start = Instant::now());
	}

	fn exit(&self, span: &Id) {
		let spans = &mut *self.spans.lock();
		spans.entry(span.clone()).and_modify(|data| {
			data.total += Instant::now() - data.start;
		});
	}

	fn try_close(&self, id: Id) -> bool {
		let spans = &mut *self.spans.lock();

		let Some((_, data)) = spans.remove_entry(&id) else {
			return true;
		};

		let json = json!({"duration": data.total.as_nanos()});
		to_writer(&mut *self.file.lock(), &json).unwrap();
		self.file.lock().write_all(b"\n").unwrap();

		true
	}
}

impl Drop for Tracer {
	fn drop(&mut self) {
		self.file.get_mut().flush().unwrap();
	}
}
