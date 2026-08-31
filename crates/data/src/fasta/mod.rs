use std::fmt::{self};

use crate::seq::{Character, write_str};

mod parser;

#[cfg(feature = "python")]
pub mod python;
pub use parser::FastaParser;

#[derive(Clone, PartialEq, Eq)]
pub struct Record<C: Character> {
	/// The sequence description.  Must start with a '>' character and have
	/// an ID follow right after without a space.
	raw_description: String,
	seq: Vec<C>,
}

impl<C: Character> Record<C> {
	pub fn new(raw_description: String, seq: Vec<C>) -> Self {
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
		let capacity = self.raw_description.len()
			+ 1 + self.sequence().len()
			+ self.sequence().len() / 80
			+ 1;
		let mut out = String::with_capacity(capacity);

		out.push_str(self.raw_description());
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
