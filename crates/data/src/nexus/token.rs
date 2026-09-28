use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TokenKind {
	Word(String),
	Punctuation(char),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Token {
	pub kind: TokenKind,
	pub start: usize,
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

	fn next_token(&mut self) -> Result<Option<Token>> {
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
				start,
				end: self.offset,
			}));
		}
		if matches!(character, '\'' | '"') {
			return self.quoted(character, start).map(Some);
		}

		let mut value = String::new();
		while let Some(character) = self.current() {
			if character == '[' {
				self.comment()?;
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
			value.push(character);
			self.offset += character.len_utf8();
		}

		Ok(Some(Token {
			kind: TokenKind::Word(value),
			start,
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

	fn quoted(&mut self, quote: char, start: usize) -> Result<Token> {
		self.offset += quote.len_utf8();
		let mut value = String::new();
		while let Some(character) = self.current() {
			self.offset += character.len_utf8();
			if character != quote {
				value.push(character);
				continue;
			}
			if self.current() == Some(quote) {
				value.push(quote);
				self.offset += quote.len_utf8();
				continue;
			}
			return Ok(Token {
				kind: TokenKind::Word(value),
				start,
				end: self.offset,
			});
		}
		bail!("Unterminated quoted token starting at byte {start}")
	}

	fn current(&self) -> Option<char> {
		self.input[self.offset..].chars().next()
	}
}

impl Iterator for Tokens<'_> {
	type Item = Result<Token>;

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

	use super::{TokenKind, Tokens};

	fn kinds(input: &str) -> Result<Vec<TokenKind>> {
		Tokens::new(input).map(|token| Ok(token?.kind)).collect()
	}

	#[test]
	fn reads_words_quotes_punctuation_and_comments() -> Result<()> {
		assert_eq!(
			kinds(" BE[ignored]GIN 'tree'' name' = (1,2); ")?,
			[
				TokenKind::Word("BEGIN".to_owned()),
				TokenKind::Word("tree' name".to_owned()),
				TokenKind::Punctuation('='),
				TokenKind::Punctuation('('),
				TokenKind::Word("1".to_owned()),
				TokenKind::Punctuation(','),
				TokenKind::Word("2".to_owned()),
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
}
