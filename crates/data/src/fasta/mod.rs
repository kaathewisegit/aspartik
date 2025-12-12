use anyhow::{Context, Error, Result, anyhow};

use std::{
	cmp::min,
	fmt::{self, Pointer},
	mem,
};

use crate::seq::{Character, Sequence, SequenceMut, parse_append_str};

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record<C: Character> {
	/// The sequence description.  Must start with a '>' character and have
	/// an ID follow right after without a space.
	raw_description: String,
	seq: Sequence<C>,
}

impl<C: Character> Record<C> {
	pub fn new(raw_description: String, seq: Sequence<C>) -> Self {
		Self {
			raw_description,
			seq,
		}
	}

	/// The sequence header line, exactly as it appeared in the source.
	pub fn raw_description(&self) -> &str {
		&self.raw_description
	}

	/// Description, excludes the starting '>'.
	pub fn description(&self) -> &str {
		// SAFETY: this won't panic because `description` must start
		// with an ASCII character '>'.
		&self.raw_description[1..]
	}

	pub fn id(&self) -> &str {
		// The FASTA identifier is the part of the description until the
		// first space.  This gets the index of the first space or
		// returns the description whole if there isn't a post-space
		// comment
		let end = self
			.raw_description
			.find(' ')
			.unwrap_or(self.raw_description.len());

		&self.raw_description[1..end]
	}

	pub fn sequence(&self) -> &Sequence<C> {
		&self.seq
	}

	pub fn into_sequence(self) -> Sequence<C> {
		self.seq
	}

	pub fn len(&self) -> usize {
		self.seq.len()
	}

	pub fn is_empty(&self) -> bool {
		self.seq.is_empty()
	}
}

impl<C: Character> fmt::Display for Record<C> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(self.raw_description())?;
		f.write_str("\n")?;

		const LINE_LEN: usize = 80;
		let seq_len = self.seq.len();
		let num_lines = seq_len.div_ceil(LINE_LEN);
		for i in 0..num_lines {
			let end = min(seq_len, (i + 1) * LINE_LEN);
			let slice = &self.seq.as_ref()[(i * LINE_LEN)..end];
			slice.fmt(f)?;
		}

		Ok(())
	}
}

#[derive(Debug)]
pub struct FastaParser<C: Character> {
	/// Since sequence descriptions must start with a '>' character,
	/// `description` being empty must mean that we haven't read the first
	/// record yet.
	description: String,
	chars: SequenceMut<C>,
	line_idx: usize,
}

impl<C: Character> FastaParser<C> {
	pub fn new() -> Self {
		// XXX: Default trait?
		Self {
			description: String::new(),
			chars: SequenceMut::new(),
			line_idx: 0,
		}
	}

	/// Takes the values and turns them into a [`Record`]
	fn make_record(&mut self) -> Option<Record<C>> {
		let description = mem::take(&mut self.description);

		if description.is_empty() {
			return None;
		}

		let chars = mem::take(&mut self.chars);

		Some(Record {
			raw_description: description,
			seq: chars.into_sequence(),
		})
	}

	/// Incrementally parse a FASTA file
	///
	/// Returns `None` if more lines are needed.  `line` being `None` is
	/// taken as an EOF.
	pub fn read_line(
		&mut self,
		line: Option<&str>,
	) -> Result<Option<Record<C>>> {
		let Some(line) = line else {
			return Ok(self.make_record());
		};

		self.line_idx += 1;

		// skip comments and empty lines
		if line.starts_with(";") || line.trim().is_empty() {
			return Ok(None);
		}

		if line.starts_with(">") {
			let out = self.make_record();
			self.description = line.to_owned();
			self.chars = SequenceMut::new();
			return Ok(out);
		}

		if self.description.is_empty() {
			return Err(anyhow!(
				"Encountered a sequence which does not belong to a record:\n{}: {}",
				self.line_idx,
				line
			));
		}

		parse_append_str(&mut self.chars, line)
			.with_context(|| sequence_error(self))?;

		Ok(None)
	}
}

impl<C: Character> Default for FastaParser<C> {
	fn default() -> Self {
		Self::new()
	}
}

fn sequence_error<C: Character>(fasta: &FastaParser<C>) -> Error {
	let record = if fasta.description.is_empty() {
		String::new()
	} else {
		format!(" for the record '{}'", fasta.description)
	};

	anyhow!(
		"Failed to parse sequence{} at line {}",
		record,
		fasta.line_idx,
	)
}
