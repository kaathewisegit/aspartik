use anyhow::{Result, anyhow, ensure};

use std::{collections::HashSet, io::BufRead};

use crate::TaxonSet;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
	Outside,
	Trees,
	Other,
}

enum TranslationDelimiter {
	Comma,
	Semicolon,
	NextLine,
}

struct NexusParser {
	block: Block,
	translation: Option<(TaxonSet, TaxonSet)>,
	pending_translation: Option<Vec<(String, String)>>,
	awaiting_translation_semicolon: bool,
	saw_tree: bool,
}

impl Default for NexusParser {
	fn default() -> Self {
		Self {
			block: Block::Outside,
			translation: None,
			pending_translation: None,
			awaiting_translation_semicolon: false,
			saw_tree: false,
		}
	}
}

fn after_keyword<'a>(line: &'a str, keyword: &str) -> Option<&'a str> {
	let head = line.get(..keyword.len())?;
	let rest = &line[keyword.len()..];
	(head.eq_ignore_ascii_case(keyword)
		&& rest.chars().next().is_none_or(char::is_whitespace))
	.then_some(rest.trim_start())
}

fn translation_entry(
	pairs: &mut Vec<(String, String)>,
	line: &str,
) -> Result<TranslationDelimiter> {
	let (entry, delimiter) = match line.as_bytes().last() {
		Some(b',') => {
			(&line[..line.len() - 1], TranslationDelimiter::Comma)
		}
		Some(b';') => (
			&line[..line.len() - 1],
			TranslationDelimiter::Semicolon,
		),
		_ => (line, TranslationDelimiter::NextLine),
	};
	let entry = entry.trim();
	let split = entry
		.find(char::is_whitespace)
		.ok_or_else(|| anyhow!("Expected a taxon name in TRANSLATE"))?;
	let alias = &entry[..split];
	let name = entry[split..].trim();
	ensure!(
		!alias.is_empty() && !name.is_empty(),
		"Expected a TRANSLATE pair"
	);
	ensure!(
		!alias.contains(['[', ']']) && !name.contains(['[', ']']),
		"Comments inside TRANSLATE entries are unsupported"
	);
	let name = if name.starts_with('\'') {
		ensure!(
			name.len() > 1 && name.ends_with('\''),
			"Unterminated quoted taxon name"
		);
		name[1..name.len() - 1].replace("''", "'")
	} else {
		ensure!(
			!name.chars().any(char::is_whitespace),
			"Unquoted taxon name contains whitespace"
		);
		name.to_owned()
	};
	pairs.push((alias.to_owned(), name));
	Ok(delimiter)
}

fn translation_sets(
	mut pairs: Vec<(String, String)>,
) -> Result<(TaxonSet, TaxonSet)> {
	ensure!(!pairs.is_empty(), "Empty TRANSLATE command");
	let mut aliases = HashSet::new();
	let mut taxa = HashSet::new();
	for (alias, name) in &pairs {
		ensure!(
			aliases.insert(alias),
			"Duplicate translation key: {alias}"
		);
		ensure!(taxa.insert(name), "Duplicate taxon name: {name}");
	}
	pairs.sort_unstable_by(|a, b| a.1.cmp(&b.1));
	Ok((
		TaxonSet::from_iter(pairs.iter().map(|(alias, _)| alias)),
		TaxonSet::from_iter(pairs.iter().map(|(_, name)| name)),
	))
}

impl NexusParser {
	fn translation_line(&mut self, text: &str) -> Result<()> {
		if self.awaiting_translation_semicolon {
			ensure!(
				text == ";",
				"Expected ';' after TRANSLATE entries"
			);
			self.awaiting_translation_semicolon = false;
			self.translation = Some(translation_sets(
				self.pending_translation.take().unwrap(),
			)?);
			return Ok(());
		}
		let pairs = self.pending_translation.as_mut().unwrap();
		match translation_entry(pairs, text)? {
			TranslationDelimiter::Comma => {}
			TranslationDelimiter::Semicolon => {
				self.translation = Some(translation_sets(
					self.pending_translation
						.take()
						.unwrap(),
				)?);
			}
			TranslationDelimiter::NextLine => {
				self.awaiting_translation_semicolon = true;
			}
		}
		Ok(())
	}

	fn line<F>(&mut self, text: &str, callback: &mut F) -> Result<()>
	where
		F: FnMut(&str, Option<(&TaxonSet, &TaxonSet)>) -> Result<()>,
	{
		if text.is_empty()
			|| text.starts_with('[')
			|| text.eq_ignore_ascii_case("#NEXUS")
		{
			return Ok(());
		}
		if self.pending_translation.is_some() {
			return self.translation_line(text);
		}
		if self.block == Block::Outside {
			if let Some(rest) = after_keyword(text, "BEGIN") {
				self.block = if rest
					.strip_suffix(';')
					.is_some_and(|name| {
						name.trim()
							.eq_ignore_ascii_case(
								"TREES",
							)
					}) {
					Block::Trees
				} else {
					Block::Other
				};
				self.translation = None;
				self.saw_tree = false;
			}
			return Ok(());
		}
		if text.eq_ignore_ascii_case("END;")
			|| text.eq_ignore_ascii_case("ENDBLOCK;")
		{
			self.block = Block::Outside;
			self.translation = None;
			return Ok(());
		}
		if self.block == Block::Other {
			return Ok(());
		}
		if let Some(rest) = after_keyword(text, "TRANSLATE") {
			ensure!(
				!self.saw_tree && self.translation.is_none(),
				"Unexpected TRANSLATE command"
			);
			self.pending_translation = Some(Vec::new());
			if !rest.is_empty() {
				self.translation_line(rest)?;
			}
			return Ok(());
		}
		if let Some(rest) = after_keyword(text, "TREE")
			.or_else(|| after_keyword(text, "UTREE"))
		{
			let rest = rest
				.strip_prefix('*')
				.unwrap_or(rest)
				.trim_start();
			let (name, newick) =
				rest.split_once('=').ok_or_else(|| {
					anyhow!("Expected '=' in TREE command")
				})?;
			let newick = newick.trim();
			ensure!(
				!name.trim().is_empty(),
				"Expected a tree name"
			);
			ensure!(
				newick.ends_with(';'),
				"Unterminated TREE command"
			);
			self.saw_tree = true;
			callback(
				newick,
				self.translation.as_ref().map(|(a, t)| (a, t)),
			)?;
		}
		Ok(())
	}

	fn finish(self) -> Result<()> {
		ensure!(
			self.pending_translation.is_none(),
			"Unterminated TRANSLATE command"
		);
		ensure!(
			self.block == Block::Outside,
			"Unterminated NEXUS block"
		);
		Ok(())
	}
}

pub fn for_each_tree<R, F>(mut reader: R, mut callback: F) -> Result<()>
where
	R: BufRead,
	F: FnMut(&str, Option<(&TaxonSet, &TaxonSet)>) -> Result<()>,
{
	let mut parser = NexusParser::default();
	let mut line = String::new();
	while reader.read_line(&mut line)? != 0 {
		parser.line(line.trim(), &mut callback)?;
		line.clear();
	}
	parser.finish()
}
