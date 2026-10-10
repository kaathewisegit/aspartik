use anyhow::{Result, anyhow, bail, ensure};

#[cfg(feature = "python")]
pub mod python;
mod trees;

pub use trees::read_binary_trees;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Status {
	#[default]
	BeforeStart,
	OutsideBlock,
	InBlock,
	InCommand,
}

#[derive(Debug, Clone, Default)]
pub struct NexusParser {
	status: Status,
	/// Block name, always in uppercase
	block: String,
	command: String,
	comment_depth: usize,
	// XXX: line tracking
}

fn read_command<'a>(line: &mut &'a str) -> Result<&'a str> {
	let end = line
		.find(|c: char| !c.is_ascii_alphanumeric())
		.unwrap_or(line.len());
	let cmd = &line[..end];
	*line = &line[end..];
	ensure!(
		end > 0,
		"Expected a command token, got trailing characters {line:?}"
	);
	Ok(cmd)
}

pub(crate) fn parse_token<'a>(line: &mut &'a str) -> Result<&'a str> {
	*line = line.trim_ascii_start();

	let first = line.chars().next().ok_or_else(|| {
		anyhow!("Expected a name, reached end of input")
	})?;

	if first == '"' || first == '\'' {
		let body = &line[1..];
		let end = body.find(first).ok_or_else(|| {
			anyhow!("Unterminated quoted name: {line:?}")
		})?;
		*line = &body[end + 1..];
		Ok(&body[..end])
	} else {
		let end = line
			.find(|c: char| c.is_ascii_whitespace() || c == ',')
			.unwrap_or(line.len());
		let name = &line[..end];
		*line = &line[end..];
		Ok(name)
	}
}

impl NexusParser {
	fn skip_comment(&mut self, line: &mut &str) {
		if self.comment_depth == 0 && !line.starts_with('[') {
			return;
		}
		for (i, c) in line.char_indices() {
			match c {
				'[' => self.comment_depth += 1,
				']' => self.comment_depth -= 1,
				_ => {}
			}
			if self.comment_depth == 0 {
				*line = &line[i + c.len_utf8()..];
				return;
			}
		}

		*line = "";
	}

	fn skip_trivia(&mut self, line: &mut &str) {
		loop {
			let prev = *line;
			*line = line.trim_ascii_start();
			self.skip_comment(line);

			if *line == prev {
				break;
			}
		}
	}

	fn expect_semi(&mut self, line: &mut &str) -> Result<()> {
		self.skip_trivia(line);
		ensure!(
			line.starts_with(";"),
			"Expected a semicolon, got {line:?}"
		);
		*line = &line[1..];
		self.skip_trivia(line);
		ensure!(
			line.is_empty(),
			"Unexpected trailing characters after semicolon: {line:?}"
		);
		Ok(())
	}

	fn parse_command<F>(&mut self, line: &mut &str, f: F) -> Result<()>
	where
		F: FnOnce(&str, &str) -> Result<()>,
	{
		let bytes = line.as_bytes();
		let mut idx = 0;

		while idx < bytes.len() {
			match bytes[idx] {
				b'[' => self.comment_depth += 1,
				b']' => {
					ensure!(
						self.comment_depth > 0,
						"Unxepected `]` outside of a comment"
					);
					self.comment_depth -= 1;
				}
				b';' if self.comment_depth == 0 => {
					self.command.push_str(&line[..=idx]);
					*line = &line[idx..];
					self.status = Status::InBlock;
					self.expect_semi(line)?;
					f(&self.block, &self.command)?;
					self.command.clear();
					return Ok(());
				}
				_ => {}
			}
			idx += 1;
		}

		// No terminating semicolon found, consume the whole line
		self.command.push_str(line);
		*line = "";
		Ok(())
	}

	pub fn parse_line<F>(
		&mut self,
		mut line: &str,
		callback: F,
	) -> Result<()>
	where
		F: FnOnce(&str, &str) -> Result<()>,
	{
		match self.status {
			Status::BeforeStart => {
				ensure!(
					line == "#NEXUS",
					"NEXUS file must start with the `#NEXUS` magic value"
				);
				self.status = Status::OutsideBlock;
			}
			Status::OutsideBlock => {
				self.skip_trivia(&mut line);
				if line.is_empty() {
					return Ok(());
				}
				let cmd = read_command(&mut line)?;
				if !cmd.eq_ignore_ascii_case("BEGIN") {
					bail!(
						"Top-level commands must start with BEGIN, got `{cmd}`"
					);
				}
				self.skip_trivia(&mut line);
				let block = read_command(&mut line)?;
				self.block = block.to_uppercase();
				self.status = Status::InBlock;
				self.expect_semi(&mut line)?;
			}
			Status::InBlock => {
				self.skip_trivia(&mut line);
				if line.is_empty() {
					return Ok(());
				}
				let cmd = read_command(&mut line)?;
				if cmd.eq_ignore_ascii_case("END") {
					self.status = Status::OutsideBlock;
					self.expect_semi(&mut line)?;
					return Ok(());
				}
				self.command.push_str(cmd);
				self.status = Status::InCommand;
				self.parse_command(&mut line, callback)?;
			}
			Status::InCommand => {
				self.parse_command(&mut line, callback)?;
			}
		}

		Ok(())
	}

	pub fn finish(&self) -> Result<()> {
		match self.status {
			Status::BeforeStart => bail!("Empty file"),
			Status::InCommand => {
				bail!("Command {:?} not finished", self.command)
			}
			Status::InBlock => {
				bail!("Block {} not finished", self.block)
			}
			Status::OutsideBlock => Ok(()),
		}
	}
}
