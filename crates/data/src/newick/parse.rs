use anyhow::Result;

use std::str::FromStr;

use super::{Edge, Node, NodeIdx, Tree};

type PNode = (Option<String>, Option<f64>);

#[derive(Debug, Clone)]
struct Subtree {
	parsed: PNode,
	children: Vec<Subtree>,
}

peg::parser! {grammar newick_parser() for str {
	rule number() -> f64 =
		number:$(
			"-"?
			("0" / (['0'..='9']+ ['0'..='9']*))
			("." ['0'..='9']+)?
			(['e' | 'E'] ("-" / "+")? ['0'..='9']+)?
		) { f64::from_str(number).unwrap() }

	rule name() -> String =
		name:$(['a'..='z' | 'A'..='Z' | '0'..='9' | '/' | '_' | '-' | '.']+)
		{ String::from(name) }
		/
		"\"" name:$([^'"']+) "\""
		{ String::from(name) }

	pub rule node() -> PNode =
		name:name()? ":" length:number()? { (name, length) }
		/ { (None, None) }

	rule subtree() -> Subtree =
		"(" children:(subtree() ++ ",") ")" parsed:node()
		{ Subtree { parsed, children } }
		/
		node:node()
		{ Subtree { parsed: node, children: Vec::new() } }

	pub rule tree() -> Subtree =
		subtree:subtree() ";" { subtree }
}}

fn add_subtree(tree: &mut Tree, subtree: Subtree) -> NodeIdx {
	let node =
		Node::new(subtree.parsed.0.unwrap_or_default(), String::new());

	let node_ref = tree.add_node(node);

	for child in subtree.children {
		let edge_length = child.parsed.1;
		let child_ref = add_subtree(tree, child);
		tree.add_edge(
			node_ref,
			child_ref,
			Edge::new(edge_length, String::new()),
		);
	}

	node_ref
}

impl Tree {
	pub fn parse(input: &str) -> Result<Self> {
		let mut out = Tree::new();

		let root = newick_parser::tree(input.trim())?;
		let root_ref = add_subtree(&mut out, root);
		out.set_root(root_ref);

		Ok(out)
	}
}
