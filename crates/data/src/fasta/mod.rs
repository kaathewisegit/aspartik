use anyhow::{Context, Result, anyhow, bail};

use std::fmt::{self};

use crate::seq::{Character, parse_append_str, write_str};

#[cfg(feature = "python")]
pub mod python;

#[derive(Clone, PartialEq, Eq)]
pub struct Record<C: Character> {
	description: String,
	seq: Vec<C>,
}

impl<C: Character> Record<C> {
	pub fn new(description: String, seq: Vec<C>) -> Self {
		Self { description, seq }
	}

	/// Description, excludes the starting '>'.
	pub fn description(&self) -> &str {
		// SAFETY: this won't panic because `description` must start
		// with an ASCII character '>'.
		&self.description
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

	pub fn sequence(&self) -> &[C] {
		&self.seq
	}

	pub fn into_sequence(self) -> Vec<C> {
		self.seq
	}

	pub fn len(&self) -> usize {
		self.seq.len()
	}

	pub fn is_empty(&self) -> bool {
		self.seq.is_empty()
	}
}

impl<C: Character> ToString for Record<C> {
	fn to_string(&self) -> String {
		// newline after description + one newline per 80 seq characters
		// + final newline
		let capacity = self.description.len()
			+ 1 + self.sequence().len()
			+ self.sequence().len() / 80
			+ 1;
		let mut out = String::with_capacity(capacity);

		out.push('>');
		out.push_str(self.description());
		out.push('\n');
		write_str(self.sequence(), &mut out);

		out
	}
}

impl<C: Character> fmt::Debug for Record<C> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Record")
			.field("description", &self.description())
			.field("sequence", &self.sequence())
			.finish()
	}
}

#[derive(Debug)]
pub struct FastaParser<C: Character> {
	in_sequence: bool,
	pub description: String,
	pub seq: Vec<C>,
}

impl<C: Character> FastaParser<C> {
	pub fn new() -> Self {
		Self {
			in_sequence: false,
			description: String::new(),
			seq: Vec::new(),
		}
	}

	pub fn parse_line<F>(&mut self, callback: F, line: &str) -> Result<()>
	where
		F: FnOnce(&mut FastaParser<C>) -> Result<()>,
	{
		// comments
		if line.trim_start().starts_with(';') {
			Ok(())
		} else if let Some(line) = line.strip_prefix('>') {
			// there was a last record before this one
			if self.in_sequence {
				self.in_sequence = false;
				callback(self)?;
			}
			// TODO: clear sequence
			self.description.clear();
			self.description.push_str(line);
			self.in_sequence = true;
			Ok(())
		} else {
			if !self.in_sequence {
				bail!(
					"Encountered sequence data before any record"
				);
			}

			parse_append_str(&mut self.seq, line).with_context(
				|| {
					anyhow!(
						"Failed to parse sequence {}",
						&self.description[1..],
					)
				},
			)
		}
	}
}

impl<C: Character> Default for FastaParser<C> {
	fn default() -> Self {
		Self::new()
	}
}
