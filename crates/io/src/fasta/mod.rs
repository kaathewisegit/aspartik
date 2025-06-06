use anyhow::{anyhow, Context, Error, Result};

use std::{
	cmp::min,
	fmt,
	fs::File,
	io::{BufRead, BufReader},
	mem,
};

use data::seq::{parse_append_str, FromChars, Seq};

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone)]
pub struct Record<S: Seq> {
	/// The sequence description.  Must start with a '>' character and have
	/// an ID follow right after without a space.
	description: String,
	seq: S,
}

impl<S: Seq> Record<S> {
	pub fn new(description: String, seq: S) -> Self {
		Self { description, seq }
	}

	/// The sequence header line, exactly as it appeared in the source.
	pub fn raw_description(&self) -> &str {
		&self.description
	}

	/// Description, excludes the starting '>'.
	pub fn description(&self) -> &str {
		// SAFETY: this won't panic because `description` must start
		// with an ASCII character '>'.
		&self.description[1..]
	}

	pub fn id(&self) -> &str {
		// The FASTA identifier is the part of the description until the
		// first space.  This gets the index of the first space or
		// returns the description whole if there isn't a post-space
		// comment
		let end = self
			.description
			.find(' ')
			.unwrap_or(self.description.len());

		&self.description[1..end]
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

pub struct FastaReader<S: Seq, R> {
	/// As sequence descriptions must start with a '>' character,
	/// `description` being empty must mean that we haven't read the first
	/// record yet.
	description: String,
	chars: Vec<S::Character>,
	reader: R,
	line_idx: usize,
}

impl<S: FromChars> FastaReader<S, BufReader<File>> {
	fn from_file(file: File) -> Self {
		let reader = BufReader::new(file);
		Self::new(reader)
	}
}

impl<S: FromChars, R: BufRead> FastaReader<S, R> {
	/// Creates a FASTA parser from a byte reader.  The reader is wrapped in
	/// `BufReader` internally, so there's no need for the caller to buffer
	/// it manually.
	pub fn new(reader: R) -> Self {
		FastaReader {
			description: String::new(),
			chars: Vec::new(),
			reader,
			line_idx: 0,
		}
	}

	fn make_record(&mut self) -> Option<Result<Record<S>>> {
		let description = mem::take(&mut self.description);

		if description.is_empty() {
			return None;
		}

		let chars = mem::take(&mut self.chars);

		let seq = S::from_vec(chars);

		Some(Ok(Record { description, seq }))
	}
}

macro_rules! bubble {
	($e:expr) => {
		match $e {
			Err(e) => return Some(Err(e.into())),
			Ok(out) => out,
		}
	};
}

impl<S: FromChars, R: BufRead> Iterator for FastaReader<S, R> {
	type Item = Result<Record<S>>;

	fn next(&mut self) -> Option<Result<Record<S>>> {
		loop {
			let mut line = String::new();
			if bubble!(self.reader.read_line(&mut line)) == 0 {
				// EOF
				return self.make_record();
			}
			trim_line_end(&mut line);
			self.line_idx += 1;

			// skip comments and empty lines
			if line.starts_with(";") || line.trim().is_empty() {
				continue;
			}

			if line.starts_with(">") {
				let out = self.make_record();
				self.description = line.to_owned();
				self.chars = Vec::new();

				if out.is_some() {
					return out;
				} else {
					continue;
				}
			}

			if self.description.is_empty() {
				return Some(Err(anyhow!("Encountered a sequence which does not belong to a record:\n{}: {}", self.line_idx, line)));
			}

			bubble!(parse_append_str(&mut self.chars, &line)
				.with_context(|| sequence_error(self)));
		}
	}
}

fn trim_line_end(line: &mut String) {
	if line.ends_with('\n') {
		line.pop();
		if line.ends_with('\r') {
			line.pop();
		}
	}
}

fn sequence_error<S: Seq, R: BufRead>(fasta: &FastaReader<S, R>) -> Error {
	if !fasta.description.is_empty() {
		anyhow!(
			"Failed to parse sequence for the record '{}' at line {}",
			fasta.description, fasta.line_idx,
		)
	} else {
		anyhow!("Failed to parse sequence at line {}", fasta.line_idx)
	}
}
