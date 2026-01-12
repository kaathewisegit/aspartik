use anyhow::{Context, Result, anyhow, bail};

use std::mem;

use super::Record;
use crate::seq::{Character, SequenceMut, parse_append_str};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
	Blank,
	InDescription,
	InSequence,
	ParsedRecord,
	Done,
}

#[derive(Debug)]
pub struct FastaParser<C: Character> {
	state: State,

	description: String,
	seq: SequenceMut<C>,
}

impl<C: Character> FastaParser<C> {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		Self {
			state: State::Blank,

			description: String::new(),
			seq: SequenceMut::new(),
		}
	}

	pub fn parse_record(
		&mut self,
		src: &mut &str,
	) -> Result<Option<Record<C>>> {
		loop {
			if src.is_empty() {
				return Ok(None);
			}

			match self.state {
				State::Blank => self.seek_description(src)?,
				State::InDescription => {
					self.read_description(src)
				}
				State::InSequence => self.read_sequence(src)?,
				State::ParsedRecord => {
					self.state = State::Blank;
					return Ok(Some(self.make_record()));
				}
				State::Done => {
					bail!("Unreachable");
				}
			}
		}
	}

	/// Takes the values and turns them into a [`Record`]
	fn make_record(&mut self) -> Record<C> {
		let raw_description = mem::take(&mut self.description);

		let seq = mem::take(&mut self.seq).into_sequence();

		Record {
			raw_description,
			seq,
		}
	}

	fn seek_description(&mut self, src: &mut &str) -> Result<()> {
		for line in src.lines() {
			if line.starts_with('>') {
				let ptr = line.as_ptr().addr();
				let src_start = src.as_ptr().addr();
				let advance = ptr - src_start;
				*src = &src[advance..];

				self.state = State::InDescription;
				return Ok(());
			} else if line.trim_start().is_empty() {
				continue;
			} else {
				bail!(
					"Expected a record description, got {line}"
				);
			}
		}

		*src = &src[src.len()..];
		Ok(())
	}

	fn read_description(&mut self, src: &mut &str) {
		if let Some(line_end) = src.find('\n') {
			// Windows style line-ending got split over chunks
			if line_end == 0 && self.description.ends_with('\r') {
				self.description.pop();
				*src = &src[1..];
			} else {
				self.description.push_str(&src[..line_end]);
				if self.description.ends_with('\r') {
					self.description.pop();
				}

				// `line_end + 1` is a valid bound because it's
				// the index of the one-byte `\n` character
				*src = &src[line_end + 1..];
			}
			self.state = State::InSequence;
		} else {
			self.description.push_str(src);
			*src = &src[src.len()..];
		}
	}

	fn read_sequence(&mut self, src: &mut &str) -> Result<()> {
		for line in src.lines() {
			if line.starts_with('>') {
				let ptr = line.as_ptr().addr();
				let src_start = src.as_ptr().addr();
				let advance = ptr - src_start;
				*src = &src[advance..];

				self.state = State::ParsedRecord;
				return Ok(());
			}

			parse_append_str(&mut self.seq, line).with_context(
				|| {
					anyhow!(
						"Failed to parse sequence {}",
						&self.description[1..],
					)
				},
			)?;
		}

		*src = &src[src.len()..];
		Ok(())
	}

	pub fn get_final_record(&mut self) -> Result<Record<C>> {
		match self.state {
			State::Blank | State::InSequence => {
				// ok
			}
			_ => bail!("Parser is in progress: {:?}", self.state),
		}
		self.state = State::Done;
		let record = self.make_record();

		Ok(record)
	}

	pub fn is_done(&self) -> bool {
		self.state == State::Done
	}
}
