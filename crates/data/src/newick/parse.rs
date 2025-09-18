#![expect(dead_code)]

use logos::{Lexer, Logos};

use super::{Edge, Node};

#[derive(Debug, PartialEq, Logos)]
#[logos(skip r"[ \t\r\n]+")]
enum Token {
	#[token("(")]
	ParenOpen,
	#[token(")")]
	ParenClose,
	#[token(":")]
	Colon,
	#[token(",")]
	Comma,
	#[token(";")]
	Semi,

	#[regex(
		r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?",
		|lex| lex.slice().parse::<f64>().unwrap(),
		priority = 3,
	)]
	Number(f64),

	// TODO: alphanum
	#[regex(
		r#"[0-9a-zA-Z_-]+|"([^"]|\")*""#,
		|lex| lex.slice().to_owned(),
		priority = 2,
	)]
	Name(String),

	#[regex(r"\[[^]]*\]")]
	Comment,
}

fn node(lexer: &mut Lexer<'_, Token>) -> Node {
	let token = lexer.next().unwrap().unwrap();

	let name = match token {
		Token::Name(n) => n,
		_ => String::new(),
	};

	Node::new(name, String::new())
}

fn distance(lexer: &mut Lexer<'_, Token>) -> Edge {
	let token = lexer.next().unwrap().unwrap();
	let distance = match token {
		Token::Number(distance) => Some(distance),
		_ => None,
	};

	Edge::new(distance, String::new())
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_node() {
		assert_eq!(
			node(&mut Lexer::new("node:1.1")),
			Node::new("node".to_owned(), String::new())
		);
		assert_eq!(
			node(&mut Lexer::new(":1.1")),
			Node::new(String::new(), String::new())
		);
		assert_eq!(
			// XXX: doesn't handle EOF
			node(&mut Lexer::new("Name:,")),
			Node::new("Name".to_owned(), String::new())
		);
	}
}
