use std::fmt::{Formatter, Result as FmtResult, Write};

use super::Character;

pub fn write_str<C: Character>(seq: &[C], s: &mut String) {
	s.reserve(seq.len());

	// SAFETY: we well only push valid ASCII to the end of `s_bytes`, so
	// it'll remain valid UTF-8
	let s_bytes = unsafe { s.as_mut_vec() };

	for character in seq {
		// SAFETY: `Character.to_ascii` must return an ASCII character
		// XXX: ascii::Char stabilization
		s_bytes.push(character.to_ascii())
	}

	let _ = s_bytes;
}

/// A slower version of `write_str` for formatting only
pub fn write_fmt<C: Character>(seq: &[C], f: &mut Formatter) -> FmtResult {
	for character in seq {
		// PANIC: `to_ascii()` returns a valid ASCII character code, so
		// `from_u32` will never return `None`.
		let char = char::from_u32(character.to_ascii().into()).unwrap();
		f.write_char(char)?;
	}

	Ok(())
}
