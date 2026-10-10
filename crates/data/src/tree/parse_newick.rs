use anyhow::{Context, Result, anyhow, bail, ensure};

use std::collections::HashMap;

use crate::TaxonSet;
use crate::tree::{
	BinaryTree, Node,
	builder::{EdgeData, NodeData, TreeBuilder},
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
	Open,
	Close,
	Comma,
	Colon,
	Semicolon,
	Label(String),
	Comment(String),
}

struct Tokenizer<'a> {
	input: &'a str,
	offset: usize,
}

impl<'a> Tokenizer<'a> {
	fn new(input: &'a str) -> Self {
		Self { input, offset: 0 }
	}

	fn next(&mut self) -> Result<Option<Token>> {
		self.skip_whitespace();
		if self.offset == self.input.len() {
			return Ok(None);
		}

		let start = self.offset;
		let character = self.input[start..].chars().next().unwrap();
		self.offset += character.len_utf8();
		let token = match character {
			'(' => Token::Open,
			')' => Token::Close,
			',' => Token::Comma,
			':' => Token::Colon,
			';' => Token::Semicolon,
			'[' => Token::Comment(self.comment(start)?),
			']' => bail!("Unexpected ']' at byte {start}"),
			'\'' | '"' => {
				Token::Label(self.quoted(character, start)?)
			}
			_ => Token::Label(self.unquoted(start)),
		};
		Ok(Some(token))
	}

	fn skip_whitespace(&mut self) {
		while let Some(character) =
			self.input[self.offset..].chars().next()
		{
			if !character.is_whitespace() {
				break;
			}
			self.offset += character.len_utf8();
		}
	}

	fn comment(&mut self, start: usize) -> Result<String> {
		let mut depth = 1_u32;
		while self.offset < self.input.len() {
			let character = self.input[self.offset..]
				.chars()
				.next()
				.unwrap();
			self.offset += character.len_utf8();
			match character {
				'[' => depth += 1,
				']' => {
					depth -= 1;
					if depth == 0 {
						return Ok(self.input
							[start..self.offset]
							.to_owned());
					}
				}
				_ => {}
			}
		}
		bail!("Unterminated comment starting at byte {start}")
	}

	fn quoted(&mut self, quote: char, start: usize) -> Result<String> {
		let mut value = String::new();
		while self.offset < self.input.len() {
			let character = self.input[self.offset..]
				.chars()
				.next()
				.unwrap();
			self.offset += character.len_utf8();
			if character != quote {
				value.push(character);
				continue;
			}
			if self.input[self.offset..].starts_with(quote) {
				value.push(quote);
				self.offset += quote.len_utf8();
				continue;
			}
			return Ok(value);
		}
		bail!("Unterminated quoted label starting at byte {start}")
	}

	fn unquoted(&mut self, start: usize) -> String {
		while self.offset < self.input.len() {
			let character = self.input[self.offset..]
				.chars()
				.next()
				.unwrap();
			if character.is_whitespace()
				|| matches!(
					character,
					'(' | ')'
						| '[' | ']' | ',' | ':' | ';' | '\''
						| '"'
				) {
				break;
			}
			self.offset += character.len_utf8();
		}
		self.input[start..self.offset].to_owned()
	}
}

#[derive(Debug)]
struct ParsedNode {
	data: NodeData,
	edge: EdgeData,
	hybrid: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delimiter {
	Comma,
	Close,
	Semicolon,
}

#[derive(Debug)]
struct Frame {
	node: Node,
	children: usize,
}

pub fn parse(input: &str) -> Result<TreeBuilder> {
	let mut tokenizer = Tokenizer::new(input);
	let mut leading_attributes = String::new();
	let first = loop {
		match tokenizer.next()?.context("Expected a Newick tree")? {
			Token::Comment(value) => {
				leading_attributes.push_str(&value)
			}
			token => break token,
		}
	};
	let mut tree = TreeBuilder::new();
	let mut hybrids = HashMap::<String, Node>::new();

	if first != Token::Open {
		let (mut parsed, delimiter) =
			parse_node(first, &mut tokenizer)?;
		parsed.data.attributes.insert_str(0, &leading_attributes);
		ensure!(
			delimiter == Delimiter::Semicolon,
			"Expected ';' after root node"
		);
		let root = tree.root();
		set_node(&mut tree, root, parsed, true, &mut hybrids)?;
		ensure!(
			tokenizer.next()?.is_none(),
			"Unexpected input after ';'"
		);
		return Ok(tree);
	}

	let mut stack = vec![Frame {
		node: tree.root(),
		children: 0,
	}];
	let mut delimiter = None;

	loop {
		if let Some(current) = delimiter.take() {
			match current {
				Delimiter::Comma => parse_child(
					&mut tree,
					&mut stack,
					&mut hybrids,
					&mut tokenizer,
					&mut delimiter,
				)?,
				Delimiter::Close => {
					let frame = stack
						.pop()
						.context("Unexpected ')'")?;
					ensure!(
						frame.children > 0,
						"An internal node must have at least one child"
					);
					let (mut parsed, next) =
						parse_optional_node(
							&mut tokenizer,
						)?;
					let is_root = stack.is_empty();
					if is_root {
						parsed.data
							.attributes
							.insert_str(
							0,
							&leading_attributes,
						);
					}
					set_node(
						&mut tree,
						frame.node,
						parsed,
						is_root,
						&mut hybrids,
					)?;
					delimiter = Some(next);
				}
				Delimiter::Semicolon => {
					ensure!(
						stack.is_empty(),
						"Expected ')' before ';'"
					);
					ensure!(
						tokenizer.next()?.is_none(),
						"Unexpected input after ';'"
					);
					return Ok(tree);
				}
			}
			continue;
		}

		parse_child(
			&mut tree,
			&mut stack,
			&mut hybrids,
			&mut tokenizer,
			&mut delimiter,
		)?;
	}
}

impl TreeBuilder {
	pub fn parse_newick(input: &str) -> Result<Self> {
		parse(input)
	}
}

impl BinaryTree {
	pub fn parse_newick_with_taxa(
		input: &str,
		taxa: TaxonSet,
	) -> Result<Self> {
		TreeBuilder::parse_newick(input)?.into_binary_with_taxa(taxa)
	}

	pub fn parse_newick_with_translation(
		input: &str,
		aliases: &TaxonSet,
		taxa: TaxonSet,
	) -> Result<Self> {
		TreeBuilder::parse_newick(input)?
			.into_binary_with_translation(aliases, taxa)
	}
}

fn parse_child(
	tree: &mut TreeBuilder,
	stack: &mut Vec<Frame>,
	hybrids: &mut HashMap<String, Node>,
	tokenizer: &mut Tokenizer<'_>,
	delimiter: &mut Option<Delimiter>,
) -> Result<()> {
	let token =
		tokenizer.next()?.context("Unexpected end of Newick tree")?;
	let parent =
		stack.last().context("Unexpected subtree after root")?.node;
	match token {
		Token::Open => {
			let child = tree.add_node(
				parent,
				NodeData::unnamed(),
				EdgeData::without_distance(),
			)?;
			stack.last_mut().unwrap().children += 1;
			stack.push(Frame {
				node: child,
				children: 0,
			});
		}
		Token::Close if stack.last().unwrap().children == 0 => {
			bail!("An internal node must have at least one child")
		}
		Token::Close | Token::Comma => {
			add_leaf(tree, parent, empty_node(), hybrids)?;
			stack.last_mut().unwrap().children += 1;
			*delimiter = Some(if token == Token::Close {
				Delimiter::Close
			} else {
				Delimiter::Comma
			});
		}
		Token::Semicolon => {
			bail!("Unexpected ';' inside an internal node")
		}
		first => {
			let (parsed, next) = parse_node(first, tokenizer)?;
			add_leaf(tree, parent, parsed, hybrids)?;
			stack.last_mut().unwrap().children += 1;
			*delimiter = Some(next);
		}
	}
	Ok(())
}

fn parse_optional_node(
	tokenizer: &mut Tokenizer<'_>,
) -> Result<(ParsedNode, Delimiter)> {
	let first =
		tokenizer.next()?.context("Unexpected end of Newick tree")?;
	match first {
		Token::Comma => Ok((empty_node(), Delimiter::Comma)),
		Token::Close => Ok((empty_node(), Delimiter::Close)),
		Token::Semicolon => Ok((empty_node(), Delimiter::Semicolon)),
		first => parse_node(first, tokenizer),
	}
}

fn parse_node(
	first: Token,
	tokenizer: &mut Tokenizer<'_>,
) -> Result<(ParsedNode, Delimiter)> {
	let mut label = None;
	let mut node_attributes = String::new();
	let mut edge_attributes = String::new();
	let mut length = None;
	let mut after_colon = false;
	let mut current = Some(first);

	loop {
		let token = match current.take() {
			Some(token) => token,
			None => tokenizer
				.next()?
				.context("Unexpected end of Newick tree")?,
		};
		match token {
			Token::Comma => {
				return finish_node(
					label,
					node_attributes,
					length,
					edge_attributes,
					Delimiter::Comma,
				);
			}
			Token::Close => {
				return finish_node(
					label,
					node_attributes,
					length,
					edge_attributes,
					Delimiter::Close,
				);
			}
			Token::Semicolon => {
				return finish_node(
					label,
					node_attributes,
					length,
					edge_attributes,
					Delimiter::Semicolon,
				);
			}
			Token::Colon => {
				ensure!(
					!after_colon,
					"A node contains more than one ':'"
				);
				after_colon = true;
			}
			Token::Label(value) if after_colon => {
				ensure!(
					length.is_none(),
					"An edge contains more than one length"
				);
				length = Some(value.parse::<f64>().map_err(
					|_| {
						anyhow!(
							"Invalid branch length '{value}'"
						)
					},
				)?);
			}
			Token::Label(value) => {
				ensure!(
					label.replace(value).is_none(),
					"A node contains more than one label"
				);
			}
			Token::Comment(value) if after_colon => {
				edge_attributes.push_str(&value);
			}
			Token::Comment(value) => {
				node_attributes.push_str(&value)
			}
			Token::Open => bail!("Unexpected '(' in node data"),
		}
	}
}

fn finish_node(
	label: Option<String>,
	node_attributes: String,
	length: Option<f64>,
	edge_attributes: String,
	delimiter: Delimiter,
) -> Result<(ParsedNode, Delimiter)> {
	let name = label.unwrap_or_default();
	let hybrid = hybrid_identifier(&name)?;
	Ok((
		ParsedNode {
			data: NodeData::new(name, node_attributes),
			edge: EdgeData::new(length, edge_attributes),
			hybrid,
		},
		delimiter,
	))
}

fn empty_node() -> ParsedNode {
	ParsedNode {
		data: NodeData::unnamed(),
		edge: EdgeData::without_distance(),
		hybrid: None,
	}
}

fn hybrid_identifier(name: &str) -> Result<Option<String>> {
	let Some(index) = name.rfind('#') else {
		return Ok(None);
	};
	let identifier = &name[index + 1..];
	ensure!(
		!identifier.is_empty(),
		"A hybrid marker requires an identifier"
	);
	ensure!(
		identifier.chars().all(|character| {
			character.is_ascii_alphanumeric() || character == '_'
		}) && identifier
			.chars()
			.any(|character| character.is_ascii_digit()),
		"Invalid hybrid identifier '#{identifier}'"
	);
	Ok(Some(identifier.to_owned()))
}

fn add_leaf(
	tree: &mut TreeBuilder,
	parent: Node,
	parsed: ParsedNode,
	hybrids: &mut HashMap<String, Node>,
) -> Result<()> {
	if let Some(identifier) = &parsed.hybrid
		&& let Some(&node) = hybrids.get(identifier)
	{
		ensure!(
			parsed.data.attributes.is_empty(),
			"A hybrid reference cannot replace node metadata"
		);
		tree.add_hybrid_edge(parent, node)?;
		return Ok(());
	}

	let identifier = parsed.hybrid.clone();
	let node = tree.add_node(parent, parsed.data, parsed.edge)?;
	if let Some(identifier) = identifier {
		hybrids.insert(identifier, node);
	}
	Ok(())
}

fn set_node(
	tree: &mut TreeBuilder,
	node: Node,
	parsed: ParsedNode,
	is_root: bool,
	hybrids: &mut HashMap<String, Node>,
) -> Result<()> {
	if let Some(identifier) = &parsed.hybrid {
		ensure!(
			!hybrids.contains_key(identifier),
			"Hybrid node '#{identifier}' is defined more than once"
		);
		hybrids.insert(identifier.clone(), node);
	}
	*tree.node_mut(node) = parsed.data;
	if !is_root {
		tree.replace_edge(node, parsed.edge)?;
	}
	Ok(())
}
