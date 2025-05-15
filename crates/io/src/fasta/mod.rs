use anyhow::{anyhow, Context, Error, Result};

use std::io::{BufRead, BufReader, Lines, Read};

use data::seq::{Character, Seq};

#[cfg(feature = "python")]
mod python;

#[derive(Debug, Clone)]
pub struct Sequence<C: Character> {
	/// The sequence description.  Must start with a '>' character and have
	/// an ID follow right after without a space.
	description: String,
	seq: Seq<C>,
}

impl<C: Character> Sequence<C> {
	/// The sequence header line, exactly as it appeared in the source.
	pub fn raw_description(&self) -> &str {
		&self.description
	}

	/// Sequence description, excludes the starting '>'.
	pub fn description(&self) -> &str {
		// SAFETY: this won't panic because `description` must start
		// with an ASCII character '>'.
		&self.description[1..]
	}

	pub fn sequence(&self) -> &Seq<C> {
		&self.seq
	}

	pub fn into_sequence(self) -> Seq<C> {
		self.seq
	}
}

impl<C: Character> From<Sequence<C>> for Seq<C> {
	fn from(value: Sequence<C>) -> Self {
		value.seq
	}
}

pub struct FastaReader<C: Character, R: Read> {
	current: Option<Sequence<C>>,
	reader: Lines<BufReader<R>>,
	line: usize,
}

impl<C: Character, R: Read> FastaReader<C, R> {
	/// Creates a FASTA parser from a byte reader.  The reader is wrapped in
	/// `BufReader` internally, so there's no need for the caller to buffer
	/// it manually.
	pub fn new(reader: R) -> Self {
		FastaReader {
			current: None,
			reader: BufReader::new(reader).lines(),
			line: 0,
		}
	}
}

impl<C: Character, R: Read> Iterator for FastaReader<C, R> {
	type Item = Result<Sequence<C>>;

	fn next(&mut self) -> Option<Result<Sequence<C>>> {
		loop {
			let Some(line) = self.reader.next() else {
				return self.current.take().map(Ok);
			};
			let line = match line {
				Ok(line) => line,
				Err(err) => {
					return Some(Err(err.into()));
				}
			};
			self.line += 1;

			// skip comments and empty lines
			if line.starts_with(";") || line.trim().is_empty() {
				continue;
			}

			if line.starts_with(">") {
				let out = self.current.take();

				self.current = Some(Sequence {
					description: line.to_owned(),
					seq: Seq::new(),
				});

				if out.is_some() {
					return out.map(Ok);
				} else {
					continue;
				}
			}

			let seq: Seq<C> = match Seq::try_from(line.as_str()) {
				Ok(seq) => seq,
				Err(err) => {
					return Some(Err(err).with_context(
						|| sequence_error(self),
					))
				}
			};
			if let Some(sequence) = self.current.as_mut() {
				sequence.seq.append(seq)
			}
		}
	}
}

fn sequence_error<C: Character, R: Read>(fasta: &FastaReader<C, R>) -> Error {
	if let Some(record) = &fasta.current {
		anyhow!(
			"Failed to parse sequence for the record: '{}' at line {}",
			record.description(), fasta.line,
		)
	} else {
		anyhow!("Failed to parse sequence at line {}", fasta.line)
	}
}
