use anyhow::{Context, Result, anyhow};

use super::{Character, FromChars, SeqMut};

pub fn parse_append_str<S>(seq: &mut S, string: &str) -> Result<()>
where
	S: SeqMut,
{
	for (i, byte) in string.bytes().enumerate() {
		let Some(character) = S::Character::from_byte(byte) else {
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

pub fn parse_append_bytes<S>(seq: &mut S, bytes: &[u8]) -> Result<()>
where
	S: SeqMut,
{
	for b in bytes.iter().copied() {
		let character = b.try_into().with_context(|| {
			anyhow!("Illegal byte encodign encountered: {:#x}", b)
		})?;
		seq.push(character);
	}
	Ok(())
}

pub fn parse_str<S>(string: &str) -> Result<S>
where
	S: FromChars,
{
	let mut chars = Vec::with_capacity(string.len());

	parse_append_str(&mut chars, string)?;

	Ok(S::from_vec(chars))
}

pub fn parse_bytes<S>(bytes: &[u8]) -> Result<S>
where
	S: FromChars,
{
	let mut chars = Vec::with_capacity(bytes.len());

	parse_append_bytes(&mut chars, bytes)?;

	Ok(S::from_vec(chars))
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
