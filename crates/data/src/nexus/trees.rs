use anyhow::{Result, anyhow};

use picoarrow::array::{ArrayUtf8, NonNullable};
use std::io::BufRead;

use super::NexusParser;
use crate::{TaxonSet, tree::BinaryTree};

fn parse_translate(command: &str) -> Result<(TaxonSet, TaxonSet)> {
	let mut from = ArrayUtf8::<NonNullable>::new();
	let mut to = ArrayUtf8::<NonNullable>::new();

	let body = &command[9..command.len() - 1];

	for entry in body.split(',') {
		let entry = entry.trim();
		if entry.is_empty() {
			continue;
		}

		let mut tokens = entry.split_whitespace();
		let first = tokens
			.next()
			.ok_or_else(|| anyhow!("Missing first identifier"))?;
		let second = tokens
			.next()
			.ok_or_else(|| anyhow!("Missing second identifier"))?;

		from.push(first)?;
		to.push(second)?;
		println!("{first:?}/{second:?}");
	}

	Ok((to.into(), from.into()))
}

pub fn read_binary_trees<R>(mut reader: R) -> Result<Vec<BinaryTree>>
where
	R: BufRead,
{
	let mut out = vec![];
	let mut line = String::new();
	let mut parser = NexusParser::default();
	let mut translation: Option<(TaxonSet, TaxonSet)> = None;

	let mut callback = |block: &str, command: &str| {
		if block != "TREES" {
			return Ok(());
		}
		if command.get(..9).is_some_and(|prefix| {
			prefix.eq_ignore_ascii_case("translate")
		}) {
			translation = Some(parse_translate(command)?);
		}

		println!("command {command:?}");
		if command.get(..4).is_some_and(|prefix| {
			prefix.eq_ignore_ascii_case("tree")
		}) {
			let idx = command.find('(').unwrap();
			let tree =
				if let Some(translation) = &translation {
					BinaryTree::parse_newick_with_translation(
				&command[idx..], &translation.0, translation.1.clone())?
				} else {
					unimplemented!();
				};
			out.push(tree);
		}
		Ok(())
	};

	while reader.read_line(&mut line)? > 0 {
		if line.ends_with('\n') {
			line.pop();
			if line.ends_with('\r') {
				line.pop();
			}
		}
		parser.parse_line(&line, &mut callback)?;
		line.clear();
	}

	Ok(out)
}
