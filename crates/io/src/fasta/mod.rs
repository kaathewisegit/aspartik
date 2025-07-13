use anyhow::{anyhow, Context, Error, Result};

use std::{cmp::min, fmt, mem};

use data::seq::{parse_append_str, FromChars, Seq};

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record<S: Seq> {
	/// The sequence description.  Must start with a '>' character and have
	/// an ID follow right after without a space.
	raw_description: String,
	seq: S,
}

impl<S: Seq> Record<S> {
	pub fn new(raw_description: String, seq: S) -> Self {
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

	pub fn sequence(&self) -> &S {
		&self.seq
	}

	pub fn into_sequence(self) -> S {
		self.seq
	}
}

impl<S: Seq> fmt::Display for Record<S> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(self.raw_description())?;
		f.write_str("\n")?;

		const LINE_LEN: usize = 80;
		let seq_len = self.seq.len();
		let num_lines = seq_len.div_ceil(LINE_LEN);
		for i in 0..num_lines {
			let end = min(seq_len, (i + 1) * LINE_LEN);
			let slice = &self.seq.as_slice()[(i * LINE_LEN)..end];
			slice.fmt_impl(f)?;
		}

		Ok(())
	}
}

#[derive(Debug, Clone)]
struct FastaParser<S: Seq> {
	/// Since sequence descriptions must start with a '>' character,
	/// `description` being empty must mean that we haven't read the first
	/// record yet.
	description: String,
	chars: Vec<S::Character>,
	line_idx: usize,
}

impl<S: FromChars> FastaParser<S> {
	pub fn new() -> Self {
		// XXX: Default trait?
		Self {
			description: String::new(),
			chars: Vec::new(),
			line_idx: 0,
		}
	}

	/// Takes the values and turns them into a [`Record`]
	fn make_record(&mut self) -> Option<Record<S>> {
		let description = mem::take(&mut self.description);

		if description.is_empty() {
			return None;
		}

		let chars = mem::take(&mut self.chars);

		let seq = S::from_vec(chars);

		Some(Record {
			raw_description: description,
			seq,
		})
	}

	/// Incrementally parse a FASTA file
	///
	/// Returns `None` if more lines are needed.  `line` being `None` is
	/// taken as an EOF.
	pub fn read_line(
		&mut self,
		line: Option<&str>,
	) -> Result<Option<Record<S>>> {
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
			self.chars = Vec::new();
			return Ok(out);
		}

		if self.description.is_empty() {
			return Err(anyhow!("Encountered a sequence which does not belong to a record:\n{}: {}", self.line_idx, line));
		}

		parse_append_str(&mut self.chars, line)
			.with_context(|| sequence_error(self))?;

		Ok(None)
	}
}

fn sequence_error<S: Seq>(fasta: &FastaParser<S>) -> Error {
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
