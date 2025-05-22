use tracing::Level;

use std::env;

const LOG_VAR_NAME: &str = "ASPARTIK_LOG";

pub fn init() {
	let var = match env::var(LOG_VAR_NAME) {
		Ok(var) => var,
		Err(env::VarError::NotUnicode(_)) => {
			eprintln!("Ignoring LOG_VAR_NAME because it's not a valid Unicode string");
			return;
		}
		Err(env::VarError::NotPresent) => return,
	};

	let level = match var.as_str() {
		"error" => Level::ERROR,
		"warn" => Level::WARN,
		"info" => Level::INFO,
		"debug" => Level::DEBUG,
		"trace" => Level::TRACE,
		_ => {
			eprintln!("Ignoring LOG_VAR_NAME because it's not one of 'error', 'warn', 'info', 'debug', 'trace'");
			return;
		}
	};

	let appender = tracing_appender::rolling::minutely(".", "tracing.log");

	tracing_subscriber::fmt()
		.json()
		.with_writer(appender)
		.with_max_level(level)
		.init();
}
