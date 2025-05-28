use anyhow::{anyhow, Context, Error, Result};

use std::io::{BufRead, BufReader, Lines, Read};

use data::seq::{parse_str, SeqView};

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone)]
pub struct Record<S: SeqView> {
	/// The sequence description.  Must start with a '>' character and have
	/// an ID follow right after without a space.
	description: String,
	seq: S,
}

impl<S: SeqView> Record<S> {
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

	pub fn sequence(&self) -> &S {
		&self.seq
	}

	pub fn into_sequence(self) -> S {
		self.seq
	}
}

pub struct FastaReader<S: SeqView, R: Read> {
	current: Option<(String, Vec<S::Character>)>,
	reader: Lines<BufReader<R>>,
	line: usize,
}

impl<S: SeqView, R: Read> FastaReader<S, R> {
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

	fn take(&mut self) -> Option<Result<Record<S>>> {
		let (description, seq) = self.current.take()?;

		let seq = seq.into();

		Some(Ok(Record { description, seq }))
	}
}

impl<S: SeqView, R: Read> Iterator for FastaReader<S, R> {
	type Item = Result<Record<S>>;

	fn next(&mut self) -> Option<Result<Record<S>>> {
		loop {
			let Some(line) = self.reader.next() else {
				return self.take();
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

				self.current =
					Some((line.to_owned(), Vec::new()));

				if out.is_some() {
					return self.take();
				} else {
					continue;
				}
			}

			// XXX: allocations
			let seq: S = match parse_str(line.as_str()) {
				Ok(seq) => seq,
				Err(err) => {
					return Some(Err(err).with_context(
						|| sequence_error(self),
					))
				}
			};
			if let Some((_, ref mut sequence)) =
				self.current.as_mut()
			{
				sequence.extend_from_slice(seq.as_ref())
			}
		}
	}
}

fn sequence_error<S: SeqView, R: Read>(fasta: &FastaReader<S, R>) -> Error {
	if let Some(record) = &fasta.current {
		anyhow!(
			"Failed to parse sequence for the record '{}' at line {}",
			record.0, fasta.line,
		)
	} else {
		anyhow!("Failed to parse sequence at line {}", fasta.line)
	}
}
