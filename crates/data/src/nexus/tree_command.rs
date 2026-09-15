use anyhow::{Context, Result, bail, ensure};

use std::borrow::Cow;

use super::BlockCommand;
use super::token::{Token, TokenKind, Tokens};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeCommand {
	name: String,
	is_default: bool,
	is_rooted: Option<bool>,
	newick: String,
	line: usize,
	column: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeCommandRef<'a> {
	name: Cow<'a, str>,
	is_default: bool,
	is_rooted: Option<bool>,
	newick: &'a str,
	line: usize,
	column: usize,
}

impl TreeCommand {
	pub fn parse(command: &BlockCommand) -> Result<Self> {
		let view = TreeCommandRef::parse(
			command.block(),
			command.name(),
			command.source(),
			command.line(),
			command.column(),
		)?;
		Ok(Self {
			name: view.name.into_owned(),
			is_default: view.is_default,
			is_rooted: view.is_rooted,
			newick: view.newick.to_owned(),
			line: view.line,
			column: view.column,
		})
	}

	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn is_default(&self) -> bool {
		self.is_default
	}

	pub fn is_rooted(&self) -> Option<bool> {
		self.is_rooted
	}

	pub fn newick(&self) -> &str {
		&self.newick
	}

	pub fn line(&self) -> usize {
		self.line
	}

	pub fn column(&self) -> usize {
		self.column
	}
}

impl<'a> TreeCommandRef<'a> {
	pub fn parse(
		block: &str,
		command_name: &str,
		source: &'a str,
		line: usize,
		column: usize,
	) -> Result<Self> {
		ensure!(
			block.eq_ignore_ascii_case("trees"),
			"Expected a command from a TREES block"
		);
		let legacy_unrooted =
			if command_name.eq_ignore_ascii_case("tree") {
				false
			} else if command_name.eq_ignore_ascii_case("utree") {
				true
			} else {
				bail!("Expected a TREE or UTREE command")
			};

		let mut tokens = Tokens::new(source);
		word(&mut tokens)?.context("Expected a TREE command")?;
		let mut token = tokens
			.next()
			.transpose()?
			.context("Expected a tree name")?;
		let is_default = token.kind == TokenKind::Punctuation('*');
		if is_default {
			token = tokens
				.next()
				.transpose()?
				.context("Expected a tree name")?;
		}
		let name = match token.kind {
			TokenKind::Word(value) => value,
			TokenKind::Punctuation(character) => {
				bail!("Expected a tree name, got '{character}'")
			}
		};
		let equals =
			tokens.next().transpose()?.context("Expected '='")?;
		ensure!(
			equals.kind == TokenKind::Punctuation('='),
			"Expected '=' after the tree name"
		);

		let newick = source[equals.end..].trim_start();
		ensure!(newick != ";", "Expected a Newick tree after '='");
		let marker = rooting_marker(newick)?;
		ensure!(
			!(legacy_unrooted && marker == Some(true)),
			"UTREE conflicts with the rooted marker"
		);
		Ok(Self {
			name,
			is_default,
			is_rooted: if legacy_unrooted {
				Some(false)
			} else {
				marker
			},
			newick,
			line,
			column,
		})
	}

	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn is_default(&self) -> bool {
		self.is_default
	}

	pub fn is_rooted(&self) -> Option<bool> {
		self.is_rooted
	}

	pub fn newick(&self) -> &str {
		self.newick
	}

	pub fn line(&self) -> usize {
		self.line
	}

	pub fn column(&self) -> usize {
		self.column
	}
}

fn word<'a>(tokens: &mut Tokens<'a>) -> Result<Option<Cow<'a, str>>> {
	let Some(Token { kind, .. }) = tokens.next().transpose()? else {
		return Ok(None);
	};
	match kind {
		TokenKind::Word(value) => Ok(Some(value)),
		TokenKind::Punctuation(character) => {
			bail!("Expected a word, got '{character}'")
		}
	}
}

fn rooting_marker(newick: &str) -> Result<Option<bool>> {
	let mut offset = 0;
	let mut rooted = None;
	loop {
		while newick[offset..]
			.chars()
			.next()
			.is_some_and(char::is_whitespace)
		{
			offset += newick[offset..]
				.chars()
				.next()
				.unwrap()
				.len_utf8();
		}
		if !newick[offset..].starts_with('[') {
			return Ok(rooted);
		}

		let start = offset;
		let mut depth = 0_u32;
		for character in newick[offset..].chars() {
			offset += character.len_utf8();
			match character {
				'[' => depth += 1,
				']' => {
					depth -= 1;
					if depth == 0 {
						break;
					}
				}
				_ => {}
			}
		}
		ensure!(depth == 0, "Unterminated leading tree comment");

		let value = newick[start + 1..offset - 1].trim();
		let marker = if value.eq_ignore_ascii_case("&R") {
			Some(true)
		} else if value.eq_ignore_ascii_case("&U") {
			Some(false)
		} else {
			None
		};
		if let Some(marker) = marker {
			ensure!(
				rooted.is_none_or(|current| current == marker),
				"Conflicting rooted and unrooted markers"
			);
			rooted = Some(marker);
		}
	}
}

#[cfg(test)]
mod tests {
	use anyhow::{Context, Result};

	use std::io::Cursor;

	use super::TreeCommand;
	use crate::nexus::BlockReader;

	fn tree(input: &str) -> Result<TreeCommand> {
		let command = BlockReader::new(Cursor::new(input))?
			.next()
			.context("Expected a block command")??;
		TreeCommand::parse(&command)
	}

	#[test]
	fn parses_tree_name_default_marker_and_rooting() -> Result<()> {
		let tree = tree(
			"#NEXUS\nBEGIN TREES; TREE * 'posterior tree' = [&R] (A:1,'B; C':2); END;",
		)?;
		assert_eq!(tree.name(), "posterior tree");
		assert!(tree.is_default());
		assert_eq!(tree.is_rooted(), Some(true));
		assert_eq!(tree.newick(), "[&R] (A:1,'B; C':2);");
		assert_eq!((tree.line(), tree.column()), (2, 14));
		Ok(())
	}

	#[test]
	fn parses_unrooted_and_unspecified_trees() -> Result<()> {
		let unrooted =
			tree("#NEXUS\nBEGIN TREES; UTREE old = (A,B); END;")?;
		assert_eq!(unrooted.is_rooted(), Some(false));
		assert_eq!(unrooted.name(), "old");

		let marked =
			tree("#NEXUS\nBEGIN TREES; TREE x = [&U](A,B); END;")?;
		assert_eq!(marked.is_rooted(), Some(false));

		let unspecified =
			tree("#NEXUS\nBEGIN TREES; TREE x = (A,B); END;")?;
		assert_eq!(unspecified.is_rooted(), None);
		Ok(())
	}

	#[test]
	fn rejects_malformed_tree_commands() {
		for (input, message) in [
			(
				"#NEXUS\nBEGIN TAXA; TREE x=(A,B); END;",
				"TREES block",
			),
			("#NEXUS\nBEGIN TREES; TITLE x; END;", "TREE or UTREE"),
			(
				"#NEXUS\nBEGIN TREES; TREE = (A,B); END;",
				"tree name",
			),
			(
				"#NEXUS\nBEGIN TREES; TREE x (A,B); END;",
				"Expected '='",
			),
			(
				"#NEXUS\nBEGIN TREES; TREE x =; END;",
				"Expected a Newick tree",
			),
			(
				"#NEXUS\nBEGIN TREES; UTREE x = [&R](A,B); END;",
				"conflicts",
			),
			(
				"#NEXUS\nBEGIN TREES; TREE x = [&R][&U](A,B); END;",
				"Conflicting",
			),
		] {
			let error = format!("{:#}", tree(input).unwrap_err());
			assert!(
				error.contains(message),
				"expected {message:?} in {error:?}"
			);
		}
	}
}
