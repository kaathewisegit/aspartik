use std::io::{BufRead, BufReader, Error, ErrorKind, Read, Result};

#[derive(Debug)]
pub struct StrBufReader<R> {
	/// SAFETY: `buffer[..self.valid]` must be valid UTF-8
	buffer: BufReader<R>,
	valid: usize,
}

impl<R: Read> StrBufReader<R> {
	pub fn new(reader: R) -> Self {
		Self {
			buffer: BufReader::new(reader),
			valid: 0,
		}
	}

	pub fn fill_buf(&mut self) -> Result<&str> {
		self.buffer.fill_buf()?;
		let new_slice = &self.buffer.buffer()[self.valid..];
		if str::from_utf8(new_slice).is_ok() {
			// The whole appended chunk is valid UTF-8.  Old
			// `..valid` data is also UTF-8, so it's safe to extend
			// it.
			self.valid = self.buffer.buffer().len();
			return Ok(self.buffer());
		}

		let Some(first_chunk) = new_slice.utf8_chunks().next() else {
			// if `new_slice` is empty, `str::from_utf8` returns
			// `Ok("")` and the function exists above
			unreachable!();
		};

		let new_valid_len = first_chunk.valid().len();
		self.valid += new_valid_len;

		Ok(self.buffer())
	}

	pub fn consume(&mut self, amount: usize) -> Result<()> {
		if !self.buffer().is_char_boundary(amount) {
			let error = Error::new(
				ErrorKind::InvalidInput,
				format!(
					"Tried to consume {amount} bytes from a string buffer, but it was not at a char boundary"
				),
			);
			return Err(error);
		}

		self.buffer.consume(amount);
		self.valid -= amount;

		Ok(())
	}

	pub fn buffer(&self) -> &str {
		// SAFETY: see `self.buffer`'s safety invariant
		unsafe {
			str::from_utf8_unchecked(
				&self.buffer.buffer()[..self.valid],
			)
		}
	}
}
