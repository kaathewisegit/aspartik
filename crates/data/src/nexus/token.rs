use anyhow::{Result, bail};

use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TokenKind<'a> {
	Word(Cow<'a, str>),
	Punctuation(char),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Token<'a> {
	pub kind: TokenKind<'a>,
	pub end: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Tokens<'a> {
	input: &'a str,
	offset: usize,
}

impl<'a> Tokens<'a> {
	pub fn new(input: &'a str) -> Self {
		Self { input, offset: 0 }
	}

	fn next_token(&mut self) -> Result<Option<Token<'a>>> {
		self.skip_separators()?;
		if self.offset == self.input.len() {
			return Ok(None);
		}

		let start = self.offset;
		let character = self.current().unwrap();
		if punctuation(character) {
			self.offset += character.len_utf8();
			return Ok(Some(Token {
				kind: TokenKind::Punctuation(character),
				end: self.offset,
			}));
		}
		if matches!(character, '\'' | '"') {
			return self.quoted(character, start).map(Some);
		}

		let mut value: Option<String> = None;
		let mut segment_start = start;
		while let Some(character) = self.current() {
			if character == '[' {
				value.get_or_insert_with(String::new).push_str(
					&self.input[segment_start..self.offset],
				);
				self.comment()?;
				segment_start = self.offset;
				continue;
			}
			if character.is_whitespace()
				|| punctuation(character) || matches!(
				character,
				'\'' | '"'
			) {
				break;
			}
			if character == ']' {
				bail!("Unexpected ']' at byte {}", self.offset);
			}
			self.offset += character.len_utf8();
		}
		let value = if let Some(mut value) = value {
			value.push_str(&self.input[segment_start..self.offset]);
			Cow::Owned(value)
		} else {
			Cow::Borrowed(&self.input[start..self.offset])
		};

		Ok(Some(Token {
			kind: TokenKind::Word(value),
			end: self.offset,
		}))
	}

	fn skip_separators(&mut self) -> Result<()> {
		loop {
			while self.current().is_some_and(char::is_whitespace) {
				self.offset +=
					self.current().unwrap().len_utf8();
			}
			if self.current() != Some('[') {
				return Ok(());
			}
			self.comment()?;
		}
	}

	fn comment(&mut self) -> Result<()> {
		let start = self.offset;
		let mut depth = 0_u32;
		while let Some(character) = self.current() {
			self.offset += character.len_utf8();
			match character {
				'[' => depth += 1,
				']' => {
					depth -= 1;
					if depth == 0 {
						return Ok(());
					}
				}
				_ => {}
			}
		}
		bail!("Unterminated comment starting at byte {start}")
	}

	fn quoted(&mut self, quote: char, start: usize) -> Result<Token<'a>> {
		self.offset += quote.len_utf8();
		let content_start = self.offset;
		let mut segment_start = content_start;
		let mut value: Option<String> = None;
		while let Some(character) = self.current() {
			let position = self.offset;
			self.offset += character.len_utf8();
			if character != quote {
				continue;
			}
			if self.current() == Some(quote) {
				let value =
					value.get_or_insert_with(String::new);
				value.push_str(
					&self.input[segment_start..position],
				);
				value.push(quote);
				self.offset += quote.len_utf8();
				segment_start = self.offset;
				continue;
			}
			let value = if let Some(mut value) = value {
				value.push_str(
					&self.input[segment_start..position],
				);
				Cow::Owned(value)
			} else {
				Cow::Borrowed(
					&self.input[content_start..position],
				)
			};
			return Ok(Token {
				kind: TokenKind::Word(value),
				end: self.offset,
			});
		}
		bail!("Unterminated quoted token starting at byte {start}")
	}

	fn current(&self) -> Option<char> {
		self.input[self.offset..].chars().next()
	}
}

impl<'a> Iterator for Tokens<'a> {
	type Item = Result<Token<'a>>;

	fn next(&mut self) -> Option<Self::Item> {
		self.next_token().transpose()
	}
}

fn punctuation(character: char) -> bool {
	matches!(character, '(' | ')' | ',' | ':' | ';' | '=' | '*')
}

#[cfg(test)]
mod tests {
	use anyhow::Result;
	use std::borrow::Cow;

	use super::{TokenKind, Tokens};

	fn kinds(input: &str) -> Result<Vec<TokenKind<'_>>> {
		Tokens::new(input).map(|token| Ok(token?.kind)).collect()
	}

	#[test]
	fn reads_words_quotes_punctuation_and_comments() -> Result<()> {
		assert_eq!(
			kinds(" BE[ignored]GIN 'tree'' name' = (1,2); ")?,
			[
				TokenKind::Word("BEGIN".into()),
				TokenKind::Word("tree' name".into()),
				TokenKind::Punctuation('='),
				TokenKind::Punctuation('('),
				TokenKind::Word("1".into()),
				TokenKind::Punctuation(','),
				TokenKind::Word("2".into()),
				TokenKind::Punctuation(')'),
				TokenKind::Punctuation(';'),
			]
		);
		Ok(())
	}

	#[test]
	fn rejects_unterminated_or_unmatched_delimiters() {
		for source in ["A[broken", "'broken", "A]"] {
			assert!(kinds(source).is_err());
		}
	}

	#[test]
	fn borrows_plain_words_and_decodes_modified_words() -> Result<()> {
		let plain = Tokens::new("TREE sample =;").next().unwrap()?;
		assert!(matches!(
			plain.kind,
			TokenKind::Word(Cow::Borrowed("TREE"))
		));
		let commented = Tokens::new("sa[note]mple;").next().unwrap()?;
		assert!(
			matches!(commented.kind, TokenKind::Word(Cow::Owned(value)) if value == "sample")
		);
		let escaped = Tokens::new("'O''Brien';").next().unwrap()?;
		assert!(
			matches!(escaped.kind, TokenKind::Word(Cow::Owned(value)) if value == "O'Brien")
		);
		Ok(())
	}
}
