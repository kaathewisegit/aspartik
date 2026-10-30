use anyhow::{Result, bail};
use logos::{Lexer, Logos};

use std::fmt::{self, Formatter, Write};

use super::{Node, Tree, TreeBuilder};

fn unquote(s: &str) -> &str {
	&s[1..s.len() - 1]
}

fn write_unescaped(dest: &mut String, raw: &str) {
	dest.clear();

	let inner = &raw[1..raw.len() - 1];
	let prev = raw.chars().next().unwrap();
	let mut chars = inner.chars().peekable();
	while let Some(curr) = chars.next() {
		if curr == prev && chars.peek() == Some(&prev) {
			chars.next();
		}
		dest.push(curr);
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Logos)]
#[logos(skip(r"[ \t\r\n]+"))]
enum Token<'a> {
	#[token("(")]
	Open,
	#[token(")")]
	Close,
	#[token(",")]
	Comma,
	#[token(":")]
	Colon,
	#[token(";")]
	Semicolon,
	#[regex(r#"[^ '\"\[\t\r\n():,;]+"#, |lex| lex.slice())]
	Ident(&'a str),
	#[regex(r#"'([^']|'')*'"#, |lex| lex.slice())]
	Quoted(&'a str),
	#[regex(r"\[[^\]]*\]", |lex| unquote(lex.slice()))]
	Attribute(&'a str),
}

pub fn parse_newick<'a>(tree: &mut TreeBuilder, newick: &'a str) -> Result<()> {
	let mut lexer = Token::lexer(newick);

	let next = |lexer: &mut Lexer<'a, Token<'a>>| -> Result<Token<'a>> {
		match lexer.next() {
			Some(Ok(t)) => Ok(t),
			Some(Err(_)) => bail!("Lexing error"),
			None => bail!("Unexpected end of input"),
		}
	};

	let mut token = next(&mut lexer)?;
	if token != Token::Open {
		bail!("Expected a Newick tree to start with '('");
	}
	token = next(&mut lexer)?;

	let mut parents = Vec::new();
	let mut children = Vec::new();

	let root = tree.add_dummy_node();
	parents.push(root);
	children.push(Vec::new());
	let mut node = tree.add_dummy_node();

	loop {
		match token {
			Token::Ident(name) => {
				tree.set_node_name(node, name);
				token = next(&mut lexer)?;
			}
			Token::Quoted(name) => {
				write_unescaped(tree.name_mut(node), name);
				token = next(&mut lexer)?;
			}
			_ => {}
		}

		if let Token::Attribute(attr) = token {
			tree.set_node_attribute(node, attr);
			token = next(&mut lexer)?;
		}

		if token == Token::Colon {
			token = next(&mut lexer)?;
			if let Token::Ident(s) = token {
				tree.set_edge_length(node, s.parse::<f64>()?);
				token = next(&mut lexer)?;
			} else {
				bail!("Expected edge length");
			}

			if let Token::Attribute(attr) = token {
				tree.set_edge_attribute(node, attr);
				token = next(&mut lexer)?;
			}
		}

		match token {
			Token::Open => {
				parents.push(node);
				children.push(Vec::new());
				token = next(&mut lexer)?;
				node = tree.add_dummy_node();
			}
			Token::Comma => {
				children.last_mut().unwrap().push(node);
				token = next(&mut lexer)?;
				node = tree.add_dummy_node();
			}
			Token::Close => {
				if parents.is_empty() {
					bail!("Too many closing parentheses");
				}
				children.last_mut().unwrap().push(node);
				let parent = parents.pop().unwrap();
				let node_children = children.pop().unwrap();
				tree.set_node_children(parent, &node_children);
				for &child in &node_children {
					tree.set_node_parent(child, parent);
				}
				node = parent;
				token = next(&mut lexer)?;
			}
			Token::Semicolon => {
				if !parents.is_empty() {
					bail!("Unclosed parentheses");
				}
				if lexer.next().is_some() {
					bail!("Trailing characters");
				}
				break;
			}
			t => bail!("Unexpected token: {t:?}"),
		}
	}

	tree.set_root(root);

	Ok(())
}

pub fn serialize_newick<T: Tree + ?Sized>(
	tree: &T,
	f: &mut Formatter<'_>,
) -> fmt::Result {
	let Some(root) = tree.root() else {
		return Err(fmt::Error);
	};

	let mut parents = vec![root];
	let mut children = vec![tree.children_of(root)];

	f.write_str("(")?;

	while !parents.is_empty() {
		let Some((&first, rest)) =
			children.last_mut().unwrap().split_first()
		else {
			f.write_str(")")?;
			children.pop();
			let parent = parents.pop().unwrap();
			write_node(tree, parent, f)?;
			if !children.is_empty()
				&& !children.last().unwrap().is_empty()
			{
				f.write_str(",")?;
			}
			continue;
		};

		*children.last_mut().unwrap() = rest;

		let first_children = tree.children_of(first);
		if first_children.is_empty() {
			write_node(tree, first, f)?;
			if !rest.is_empty() {
				f.write_str(",")?;
			}
		} else {
			parents.push(first);
			children.push(first_children);
			f.write_str("(")?;
		}
	}

	f.write_str(";")?;

	Ok(())
}

fn write_node<T: Tree + ?Sized>(
	tree: &T,
	node: Node,
	f: &mut Formatter<'_>,
) -> fmt::Result {
	write_label(tree.name_of(node), f)?;
	if let Some(length) = tree.edge_length(node) {
		write!(f, ":{length}")?;
	}
	Ok(())
}

const ILLEGAL_CHARS: [char; 12] = [
	' ', '\n', '\t', '(', ')', '[', ']', ',', ':', ';', '\'', '"',
];

fn write_label(label: &str, f: &mut Formatter<'_>) -> fmt::Result {
	if label.is_empty() {
		return Ok(());
	}
	if !label.contains(ILLEGAL_CHARS) {
		return f.write_str(label);
	}

	f.write_char('\'')?;
	for character in label.chars() {
		f.write_char(character)?;
		if character == '\'' {
			f.write_char('\'')?;
		}
	}
	f.write_char('\'')
}
