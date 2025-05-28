use anyhow::{anyhow, Context, Result};

use super::SeqView;

pub fn parse_str<S>(seq: &str) -> Result<S>
where
	S: SeqView,
{
	let mut chars = Vec::with_capacity(seq.len());

	for ch in seq.chars() {
		let character = ch
			.try_into()
			.with_context(|| highlight_error(seq, chars.len()))?;
		chars.push(character);
	}

	Ok(S::from_vec(chars))
}

pub fn parse_bytes<S>(seq: &[u8]) -> Result<S>
where
	S: SeqView,
{
	let mut chars = Vec::with_capacity(seq.len());

	for b in seq.iter().copied() {
		let character = b.try_into().with_context(|| {
			anyhow!("Illegal byte encodign encountered: {:#x}", b)
		})?;
		chars.push(character);
	}

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
			src,
			"",
		)
	}
}
