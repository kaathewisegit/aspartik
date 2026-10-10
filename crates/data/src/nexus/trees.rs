use anyhow::{Result, bail};

use picoarrow::array::{ArrayUtf8, NonNullable};
use std::io::BufRead;

use super::{NexusParser, parse_token};
use crate::{TaxonSet, tree::BinaryTree};

fn parse_translate(command: &str) -> Result<(TaxonSet, TaxonSet)> {
	let mut from = ArrayUtf8::<NonNullable>::new();
	let mut to = ArrayUtf8::<NonNullable>::new();

	// cuts "translate" and ";"
	let body = &command[9..command.len() - 1];

	let mut rest = body;
	loop {
		rest = rest.trim_ascii();
		if rest.is_empty() {
			break;
		}

		let first = parse_token(&mut rest)?;
		let second = parse_token(&mut rest)?;

		from.push(first)?;
		to.push(second)?;

		rest = rest.trim_ascii_start();
		if rest.is_empty() {
			break;
		}
		if !rest.starts_with(',') {
			bail!(
				"Expected `,` between translation entries, got {rest:?}"
			);
		}
		rest = &rest[1..];
	}

	Ok((from.into(), to.into()))
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
