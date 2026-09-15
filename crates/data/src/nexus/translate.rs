use anyhow::{Context, Result, bail, ensure};

use std::collections::HashMap;

use super::BlockCommand;
use super::token::{TokenKind, Tokens};
use crate::tree::builder::TreeBuilder;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranslationTable {
	entries: HashMap<String, String>,
}

impl TranslationTable {
	pub fn parse(command: &BlockCommand) -> Result<Self> {
		Self::parse_source(
			command.block(),
			command.name(),
			command.source(),
		)
	}

	pub fn parse_source(
		block: &str,
		name: &str,
		source: &str,
	) -> Result<Self> {
		ensure!(
			block.eq_ignore_ascii_case("trees"),
			"Expected a command from a TREES block"
		);
		ensure!(
			name.eq_ignore_ascii_case("translate"),
			"Expected a TRANSLATE command"
		);

		let mut tokens = Tokens::new(source);
		required_word(&mut tokens, "a TRANSLATE command")?;
		let mut entries = HashMap::new();
		loop {
			let key = required_word(
				&mut tokens,
				"a translation key",
			)?;
			let value = required_word(&mut tokens, "a taxon name")?;
			ensure!(
				entries.insert(key.clone(), value).is_none(),
				"Translation key '{key}' appears more than once"
			);

			let delimiter = tokens
				.next()
				.transpose()?
				.context("Expected ',' or ';'")?;
			match delimiter.kind {
				TokenKind::Punctuation(',') => {}
				TokenKind::Punctuation(';') => {
					ensure!(
						tokens.next()
							.transpose()?
							.is_none(),
						"Unexpected input after ';'"
					);
					return Ok(Self { entries });
				}
				_ => bail!(
					"Expected ',' or ';' after a translation entry"
				),
			}
		}
	}

	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	pub fn get(&self, key: &str) -> Option<&str> {
		self.entries.get(key).map(String::as_str)
	}

	pub fn apply(&self, tree: &mut TreeBuilder) {
		let leaves = tree
			.nodes()
			.filter(|&node| tree.is_leaf(node))
			.collect::<Vec<_>>();
		for leaf in leaves {
			if let Some(name) = self.get(&tree.node(leaf).name) {
				tree.node_mut(leaf).name = name.to_owned();
			}
		}
	}
}

fn required_word(tokens: &mut Tokens<'_>, expected: &str) -> Result<String> {
	let Some(token) = tokens.next().transpose()? else {
		bail!("Expected {expected}");
	};
	match token.kind {
		TokenKind::Word(value) => Ok(value.into_owned()),
		TokenKind::Punctuation(character) => {
			bail!("Expected {expected}, got '{character}'")
		}
	}
}

#[cfg(test)]
mod tests {
	use anyhow::{Context, Result};

	use std::io::Cursor;

	use super::TranslationTable;
	use crate::{nexus::BlockReader, tree::parse_newick};

	fn table(input: &str) -> Result<TranslationTable> {
		let command = BlockReader::new(Cursor::new(input))?
			.next()
			.context("Expected a block command")??;
		TranslationTable::parse(&command)
	}

	#[test]
	fn parses_translation_entries() -> Result<()> {
		let table = table(
			"#NEXUS\nBEGIN TREES; TRANSLATE 1 A, 2 'B B', key[ignored]3 'O''Brien'; END;",
		)?;
		assert_eq!(table.len(), 3);
		assert!(!table.is_empty());
		assert_eq!(table.get("1"), Some("A"));
		assert_eq!(table.get("2"), Some("B B"));
		assert_eq!(table.get("key3"), Some("O'Brien"));
		assert_eq!(table.get("missing"), None);
		Ok(())
	}

	#[test]
	fn parses_borrowed_translation_source() -> Result<()> {
		let table = TranslationTable::parse_source(
			"TREES",
			"TRANSLATE",
			"TRANSLATE 1 A, 2 'B B';",
		)?;
		assert_eq!(table.get("1"), Some("A"));
		assert_eq!(table.get("2"), Some("B B"));
		Ok(())
	}

	#[test]
	fn applies_translation_only_to_leaves() -> Result<()> {
		let table = table(
			"#NEXUS\nBEGIN TREES; TRANSLATE 1 A, 2 'B B'; END;",
		)?;
		let mut tree = parse_newick("((1:1,2:2)1:3,Human:4);")?;
		table.apply(&mut tree);
		assert_eq!(tree.to_newick()?, "((A:1,'B B':2)1:3,Human:4);");
		Ok(())
	}

	#[test]
	fn rejects_malformed_translation_tables() {
		for (input, message) in [
			(
				"#NEXUS\nBEGIN TAXA; TRANSLATE 1 A; END;",
				"TREES block",
			),
			(
				"#NEXUS\nBEGIN TREES; TITLE 1 A; END;",
				"TRANSLATE command",
			),
			(
				"#NEXUS\nBEGIN TREES; TRANSLATE; END;",
				"translation key",
			),
			(
				"#NEXUS\nBEGIN TREES; TRANSLATE 1; END;",
				"taxon name",
			),
			(
				"#NEXUS\nBEGIN TREES; TRANSLATE 1 A 2 B; END;",
				"Expected ',' or ';'",
			),
			(
				"#NEXUS\nBEGIN TREES; TRANSLATE 1 A, 1 B; END;",
				"appears more than once",
			),
			(
				"#NEXUS\nBEGIN TREES; TRANSLATE 1 A,; END;",
				"translation key",
			),
		] {
			let error = format!("{:#}", table(input).unwrap_err());
			assert!(
				error.contains(message),
				"expected {message:?} in {error:?}"
			);
		}
	}
}
