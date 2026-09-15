use anyhow::{Result, bail, ensure};

use std::io::BufRead;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
	source: String,
	line: usize,
	column: usize,
}

impl Command {
	pub fn source(&self) -> &str {
		&self.source
	}

	pub fn line(&self) -> usize {
		self.line
	}

	pub fn column(&self) -> usize {
		self.column
	}
}

#[derive(Debug)]
pub struct CommandReader<R> {
	reader: R,
	line_buffer: String,
	command_buffer: String,
	offset: usize,
	line: usize,
	column: usize,
	finished: bool,
}

impl<R: BufRead> CommandReader<R> {
	pub fn new(mut reader: R) -> Result<Self> {
		let mut line_buffer = String::new();
		reader.read_line(&mut line_buffer)?;
		let mut offset = if line_buffer.starts_with('\u{feff}') {
			'\u{feff}'.len_utf8()
		} else {
			0
		};
		while let Some(character) = line_buffer[offset..].chars().next()
		{
			if !character.is_whitespace() {
				break;
			}
			offset += character.len_utf8();
		}
		let end = offset + "#NEXUS".len();
		ensure!(
			line_buffer
				.get(offset..end)
				.is_some_and(|value| value
					.eq_ignore_ascii_case("#NEXUS")),
			"Expected '#NEXUS' at the start of the file"
		);
		ensure!(
			line_buffer[end..]
				.chars()
				.next()
				.is_none_or(|character| character
					.is_whitespace()
					|| character == '['),
			"Expected whitespace after '#NEXUS'"
		);

		Ok(Self {
			reader,
			line_buffer,
			command_buffer: String::new(),
			offset: end,
			line: 1,
			column: end + 1,
			finished: false,
		})
	}

	pub fn next_command(&mut self) -> Result<Option<Command>> {
		self.next_command_with(|source, line, column| {
			Ok(Command {
				source: source.to_owned(),
				line,
				column,
			})
		})
	}

	pub fn next_command_with<T>(
		&mut self,
		callback: impl FnOnce(&str, usize, usize) -> Result<T>,
	) -> Result<Option<T>> {
		if self.finished {
			return Ok(None);
		}

		self.command_buffer.clear();
		let mut start: Option<(usize, usize)> = None;
		let mut comment_depth = 0_u32;
		let mut quote = None;
		let mut has_content = false;
		let mut callback = Some(callback);

		'input: loop {
			if self.offset == self.line_buffer.len() {
				self.line_buffer.clear();
				self.offset = 0;
			}
			if self.line_buffer.is_empty()
				&& self.reader
					.read_line(&mut self.line_buffer)? == 0
			{
				self.finished = true;
				ensure!(
					comment_depth == 0,
					"Unterminated NEXUS comment starting before line {}",
					self.line
				);
				ensure!(
					quote.is_none(),
					"Unterminated quoted NEXUS token starting before line {}",
					self.line
				);
				if !has_content {
					return Ok(None);
				}
				bail!(
					"Unterminated NEXUS command starting at line {}, column {}",
					start.map_or(self.line, |position| {
						position.0
					}),
					start.map_or(self.column, |position| {
						position.1
					})
				);
			}

			while self.offset < self.line_buffer.len() {
				let character = self.line_buffer[self.offset..]
					.chars()
					.next()
					.unwrap();
				if start.is_none() {
					if character.is_whitespace() {
						self.advance(character);
						continue;
					}
					start = Some((self.line, self.column));
				}
				self.command_buffer.push(character);
				self.advance(character);

				if comment_depth > 0 {
					match character {
						'[' => comment_depth += 1,
						']' => comment_depth -= 1,
						_ => {}
					}
					continue;
				}
				if let Some(delimiter) = quote {
					if character == delimiter {
						if self.line_buffer
							[self.offset..]
							.starts_with(delimiter)
						{
							self.command_buffer
								.push(
								delimiter,
							);
							self.advance(delimiter);
						} else {
							quote = None;
						}
					}
					continue;
				}
				if !character.is_whitespace()
					&& !matches!(character, '[' | ';')
				{
					has_content = true;
				}
				match character {
					'[' => comment_depth = 1,
					']' => bail!(
						"Unexpected ']' at line {}, column {}",
						self.line,
						self.column.saturating_sub(1)
					),
					'\'' | '"' => quote = Some(character),
					';' => {
						if !has_content {
							self.command_buffer
								.clear();
							start = None;
							continue 'input;
						}
						let (line, column) =
							start.unwrap();
						return callback
							.take()
							.unwrap()(
							&self.command_buffer,
							line,
							column,
						)
						.map(Some);
					}
					_ => {}
				}
			}
		}
	}

	fn advance(&mut self, character: char) {
		self.offset += character.len_utf8();
		if character == '\n' {
			self.line += 1;
			self.column = 1;
		} else {
			self.column += 1;
		}
	}
}

impl<R: BufRead> Iterator for CommandReader<R> {
	type Item = Result<Command>;

	fn next(&mut self) -> Option<Self::Item> {
		match self.next_command() {
			Ok(Some(command)) => Some(Ok(command)),
			Ok(None) => None,
			Err(error) => {
				self.finished = true;
				Some(Err(error))
			}
		}
	}
}
