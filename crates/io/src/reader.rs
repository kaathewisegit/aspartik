use anyhow::Result;

use std::io::{BufRead, BufReader};

use crate::rw::AnyReader;
use data::Parser;

pub struct StrReader<P> {
	parser: P,
	reader: BufReader<AnyReader>,
	finished: bool,
}

impl<P: Parser<str>> StrReader<P> {
	pub fn new(parser: P, reader: AnyReader) -> Self {
		Self {
			parser,
			reader: BufReader::new(reader),
			finished: false,
		}
	}
}

macro_rules! bubble {
	($e: expr) => {
		match $e {
			Ok(out) => out,
			Err(e) => return Some(Err(e.into())),
		}
	};
}

impl<P: Parser<str>> Iterator for StrReader<P> {
	type Item = Result<P::Output>;

	fn next(&mut self) -> Option<Self::Item> {
		while !bubble!(self.reader.fill_buf()).is_empty() {
			// XXX: string buffer which returns str
			let mut src =
				str::from_utf8(self.reader.buffer()).unwrap();
			let old_len = src.len();

			let record = bubble!(self.parser.advance(&mut src));
			let new_len = src.len();
			self.reader.consume(old_len - new_len);

			if let Some(record) = record {
				return Some(Ok(record));
			}
		}

		if self.finished {
			None
		} else {
			self.finished = true;
			self.parser.final_object().map(Ok)
		}
	}
}
