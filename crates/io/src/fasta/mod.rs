use anyhow::{anyhow, Context, Error, Result};

use std::{
	io::{BufRead, BufReader, Lines, Read},
	mem,
};

use data::seq::{parse_str, Seq, SeqView};

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
	/// As sequence descriptions must start with a '>' character,
	/// `description` being empty must mean that we haven't read the first
	/// record yet.
	description: String,
	chars: Vec<S::Character>,
	reader: Lines<BufReader<R>>,
	line: usize,
}

impl<S: SeqView, R: Read> FastaReader<S, R> {
	/// Creates a FASTA parser from a byte reader.  The reader is wrapped in
	/// `BufReader` internally, so there's no need for the caller to buffer
	/// it manually.
	pub fn new(reader: R) -> Self {
		FastaReader {
			description: String::new(),
			chars: Vec::new(),
			reader: BufReader::new(reader).lines(),
			line: 0,
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

impl<S: SeqView, R: Read> Iterator for FastaReader<S, R> {
	type Item = Result<Record<S>>;

	fn next(&mut self) -> Option<Result<Record<S>>> {
		loop {
			let Some(line) = self.reader.next() else {
				return self.make_record();
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
				return Some(Err(anyhow!("Encountered a sequence which does not belong to a record:\n{}: {}", self.line, line)));
			}

			// XXX: allocations
			type CSeq<S> =
				Seq<<S as data::seq::SeqView>::Character>;
			let seq: CSeq<S> = match parse_str(line.as_str()) {
				Ok(seq) => seq,
				Err(err) => {
					return Some(Err(err).with_context(
						|| sequence_error(self),
					))
				}
			};
			self.chars.extend_from_slice(seq.as_slice());
		}
	}
}

fn sequence_error<S: SeqView, R: Read>(fasta: &FastaReader<S, R>) -> Error {
	if !fasta.description.is_empty() {
		anyhow!(
			"Failed to parse sequence for the record '{}' at line {}",
			fasta.description, fasta.line,
		)
	} else {
		anyhow!("Failed to parse sequence at line {}", fasta.line)
	}
}
