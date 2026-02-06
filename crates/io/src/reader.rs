use anyhow::Result;

use crate::{StrBufReader, rw::AnyReader};
use data::Parser;

pub struct StrReader<P> {
	parser: P,
	reader: StrBufReader<AnyReader>,
	finished: bool,
}

impl<P: Parser<str>> StrReader<P> {
	pub fn new(parser: P, reader: AnyReader) -> Self {
		Self {
			parser,
			reader: StrBufReader::new(reader),
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
			let mut src = self.reader.buffer();
			let old_len = src.len();

			let record = bubble!(self.parser.advance(&mut src));
			let new_len = src.len();
			bubble!(self.reader.consume(old_len - new_len));

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
