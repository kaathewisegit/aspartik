use anyhow::{Error, Result, anyhow, bail};

use std::str::FromStr;

use super::{Character, Sequence, SequenceMut};

pub fn parse_append_str<C: Character>(
	seq: &mut SequenceMut<C>,
	string: &str,
) -> Result<()> {
	for (i, byte) in string.bytes().enumerate() {
		let Some(character) = C::from_ascii(byte) else {
			let ch = string[i..].chars().next().unwrap();
			return Err(anyhow!(
				"Encountered an invalid character '{ch}' in a FASTA character sequence"
			)
			.context(highlight_error(string, i)));
		};

		seq.push(character);
	}
	Ok(())
}

pub fn parse_append_bytes<C: Character>(
	seq: &mut SequenceMut<C>,
	bytes: &[u8],
) -> Result<()> {
	for b in bytes.iter().copied() {
		let Some(character) = C::from_byte(b) else {
			bail!("Illegal byte encodign encountered: {:#x}", b);
		};
		seq.push(character);
	}
	Ok(())
}

pub fn parse_str<C: Character>(string: &str) -> Result<SequenceMut<C>> {
	let mut seq = SequenceMut::with_capacity(string.len());

	parse_append_str(&mut seq, string)?;

	Ok(seq)
}

pub fn parse_bytes<C: Character>(bytes: &[u8]) -> Result<SequenceMut<C>> {
	let mut seq = SequenceMut::with_capacity(bytes.len());

	parse_append_bytes(&mut seq, bytes)?;

	Ok(seq)
}

fn highlight_error(src: &str, index: usize) -> String {
	const MAX_WIDTH: usize = 60;
	if src.len() > MAX_WIDTH {
		let mut out = String::from(
			"Illegal character encountered in the sequence:\n> ",
		);
		let mut padding = 2;

		let start = if index > 40 {
			out.push_str("...");
			padding += 3;
			index - 40
		} else {
			0
		};

		let end = std::cmp::min(start + MAX_WIDTH, src.len());
		out.push_str(&src[start..end]);
		if end < src.len() {
			out.push_str("...");
		}
		out.push('\n');
		for _ in 0..(padding + index - start) {
			out.push(' ');
		}
		out.push('^');

		out
	} else {
		format!(
			"Illegal character encountered in the sequence:\n> {}\n  {:index$}^",
			src, "",
		)
	}
}

impl<C: Character> FromStr for Sequence<C> {
	type Err = Error;

	fn from_str(s: &str) -> Result<Self> {
		Ok(parse_str(s)?.into())
	}
}
