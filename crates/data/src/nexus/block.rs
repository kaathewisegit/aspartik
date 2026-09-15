use anyhow::{Context, Result, bail, ensure};

use std::{borrow::Cow, io::BufRead};

use super::token::{TokenKind, Tokens};
use super::{Command, CommandReader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockCommand {
	block: String,
	name: String,
	command: Command,
}

impl BlockCommand {
	pub fn block(&self) -> &str {
		&self.block
	}

	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn source(&self) -> &str {
		self.command.source()
	}

	pub fn line(&self) -> usize {
		self.command.line()
	}

	pub fn column(&self) -> usize {
		self.command.column()
	}
}

#[derive(Debug)]
pub struct BlockReader<R> {
	commands: CommandReader<R>,
	block: Option<String>,
	finished: bool,
}

impl<R: BufRead> BlockReader<R> {
	pub fn new(reader: R) -> Result<Self> {
		Ok(Self {
			commands: CommandReader::new(reader)?,
			block: None,
			finished: false,
		})
	}

	pub fn next_command(&mut self) -> Result<Option<BlockCommand>> {
		if self.finished {
			return Ok(None);
		}

		loop {
			let Some(command) = self.commands.next_command()?
			else {
				self.finished = true;
				ensure!(
					self.block.is_none(),
					"NEXUS block '{}' was not closed",
					self.block
						.as_deref()
						.unwrap_or_default()
				);
				return Ok(None);
			};
			let result = self.read_command(command.clone());
			match result.with_context(|| {
				format!(
					"Invalid NEXUS command at line {}, column {}",
					command.line(),
					command.column()
				)
			})? {
				Some(command) => return Ok(Some(command)),
				None => continue,
			}
		}
	}

	pub fn next_command_with<T>(
		&mut self,
		mut callback: impl FnMut(
			&str,
			&str,
			&str,
			usize,
			usize,
		) -> Result<T>,
	) -> Result<Option<T>> {
		if self.finished {
			return Ok(None);
		}
		loop {
			let block = &mut self.block;
			let result = self.commands.next_command_with(
				|source, line, column| {
					let mut tokens = Tokens::new(source);
					let name = word_ref(&mut tokens)?
						.context(
							"Expected a command name",
						)?;
					if name.eq_ignore_ascii_case("begin") {
						ensure!(
							block.is_none(),
							"NEXUS blocks cannot be nested"
						);
						let value =
							word_ref(&mut tokens)?
								.context(
									"Expected a block name",
								)?;
						expect_end(&mut tokens)?;
						*block =
							Some(value
								.into_owned());
						return Ok(None);
					}
					if name.eq_ignore_ascii_case("end")
						|| name.eq_ignore_ascii_case(
							"endblock",
						) {
						expect_end(&mut tokens)?;
						ensure!(
							block.take().is_some(),
							"No NEXUS block is open"
						);
						return Ok(None);
					}
					let current =
						block.as_deref().context(
							"A command appears outside a NEXUS block",
						)?;
					callback(
						current, &name, source, line,
						column,
					)
					.map(Some)
				},
			);
			match result.context("Invalid NEXUS command")? {
				Some(Some(value)) => return Ok(Some(value)),
				Some(None) => continue,
				None => {
					self.finished = true;
					ensure!(
						self.block.is_none(),
						"NEXUS block '{}' was not closed",
						self.block
							.as_deref()
							.unwrap_or_default()
					);
					return Ok(None);
				}
			}
		}
	}

	fn read_command(
		&mut self,
		command: Command,
	) -> Result<Option<BlockCommand>> {
		let mut tokens = Tokens::new(command.source());
		let name = word(&mut tokens)?
			.context("Expected a command name")?;

		if name.eq_ignore_ascii_case("begin") {
			ensure!(
				self.block.is_none(),
				"NEXUS blocks cannot be nested"
			);
			let block = word(&mut tokens)?
				.context("Expected a block name")?;
			expect_end(&mut tokens)?;
			self.block = Some(block);
			return Ok(None);
		}

		if name.eq_ignore_ascii_case("end")
			|| name.eq_ignore_ascii_case("endblock")
		{
			expect_end(&mut tokens)?;
			ensure!(
				self.block.take().is_some(),
				"No NEXUS block is open"
			);
			return Ok(None);
		}

		let block = self
			.block
			.clone()
			.context("A command appears outside a NEXUS block")?;
		Ok(Some(BlockCommand {
			block,
			name,
			command,
		}))
	}
}

impl<R: BufRead> Iterator for BlockReader<R> {
	type Item = Result<BlockCommand>;

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

fn word(tokens: &mut Tokens<'_>) -> Result<Option<String>> {
	word_ref(tokens).map(|value| value.map(Cow::into_owned))
}

fn word_ref<'a>(tokens: &mut Tokens<'a>) -> Result<Option<Cow<'a, str>>> {
	let Some(token) = tokens.next().transpose()? else {
		return Ok(None);
	};
	match token.kind {
		TokenKind::Word(value) => Ok(Some(value)),
		TokenKind::Punctuation(character) => {
			bail!("Expected a word, got '{character}'")
		}
	}
}

fn expect_end(tokens: &mut Tokens<'_>) -> Result<()> {
	let token = tokens.next().transpose()?.context("Expected ';'")?;
	ensure!(token.kind == TokenKind::Punctuation(';'), "Expected ';'");
	ensure!(
		tokens.next().transpose()?.is_none(),
		"Unexpected input after ';'"
	);
	Ok(())
}

#[cfg(test)]
mod tests {
	use anyhow::Result;

	use std::io::Cursor;

	use super::BlockReader;

	fn commands(input: &str) -> Result<Vec<(String, String, String)>> {
		BlockReader::new(Cursor::new(input))?
			.map(|command| {
				let command = command?;
				Ok((
					command.block().to_owned(),
					command.name().to_owned(),
					command.source().to_owned(),
				))
			})
			.collect()
	}

	#[test]
	fn groups_commands_by_block() -> Result<()> {
		let input = "#NEXUS\nBE[ignored]GIN TAXA; DIMENSIONS NTAX=2; END;\nBEGIN 'TREES'; TREE one=(A,B); ENDBLOCK;";
		assert_eq!(
			commands(input)?,
			[
				(
					"TAXA".to_owned(),
					"DIMENSIONS".to_owned(),
					"DIMENSIONS NTAX=2;".to_owned(),
				),
				(
					"TREES".to_owned(),
					"TREE".to_owned(),
					"TREE one=(A,B);".to_owned(),
				),
			]
		);
		Ok(())
	}

	#[test]
	fn supports_multiple_and_empty_blocks() -> Result<()> {
		let input = "#NEXUS\nBEGIN NOTES; END; BEGIN TREES; TREE a=(A,B); END; BEGIN TREES; TREE b=(C,D); END;";
		let commands = commands(input)?;
		assert_eq!(commands.len(), 2);
		assert_eq!(commands[0].0, "TREES");
		assert_eq!(commands[1].0, "TREES");
		Ok(())
	}

	#[test]
	fn rejects_invalid_block_structure() {
		for (input, message) in [
			("#NEXUS\nTREE x=(A,B);", "outside a NEXUS block"),
			("#NEXUS\nBEGIN TREES;", "was not closed"),
			("#NEXUS\nEND;", "No NEXUS block is open"),
			(
				"#NEXUS\nBEGIN TREES; BEGIN TAXA; END; END;",
				"cannot be nested",
			),
			("#NEXUS\nBEGIN;", "Expected a word, got ';'"),
			("#NEXUS\nBEGIN TREES extra; END;", "Expected ';'"),
			("#NEXUS\nBEGIN TREES; END extra;", "Expected ';'"),
		] {
			let error =
				format!("{:#}", commands(input).unwrap_err());
			assert!(
				error.contains(message),
				"expected {message:?} in {error:?}"
			);
		}
	}
}
