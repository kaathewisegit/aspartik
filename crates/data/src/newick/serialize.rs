use std::fmt::{Result, Write};

use super::{Edge, Node, Tree};

impl Tree {
	pub fn serialize<W: Write>(&self, writer: &mut W) -> Result {
		serialize(self, writer)
	}

	pub fn into_string(&self) -> String {
		let mut out = String::new();
		// PANIC: writing to String is infallible
		self.serialize(&mut out).unwrap();
		out
	}
}

fn serialize<W: Write>(tree: &Tree, writer: &mut W) -> Result {
	let root = tree.root.unwrap_or(0);

	let dummy = &Edge::default();
	let mut stack = Vec::from([(root, dummy, false)]);

	macro_rules! write_comma {
		() => {
			// don't add trailing commas
			if let Some(next) = stack.last() {
				if !next.2 {
					writer.write_char(',')?;
				}
			}
		};
	}

	while let Some((node, edge, maker)) = stack.pop() {
		if maker {
			writer.write_char(')')?;
			write_node(tree.get_node(node), edge, writer)?;
			write_comma!();
			continue;
		}

		if tree.is_leaf(node) {
			// leaf
			write_node(tree.get_node(node), edge, writer)?;
			write_comma!();
		} else {
			writer.write_char('(')?;

			stack.push((node, edge, true));

			for child in tree.children_of(node).rev() {
				stack.push((
					child,
					tree.edge(node, child),
					false,
				));
			}
		}
	}

	writer.write_char(';')
}

fn write_node<W: Write>(node: &Node, parent_edge: &Edge, w: &mut W) -> Result {
	if !node.name().is_empty() {
		if node.name().contains([' ', '\'']) {
			write!(w, "\"{}\"{}", node.name(), node.attributes())?;
		} else {
			write!(w, "{}{}", node.name(), node.attributes())?;
		}
	}
	if let Some(distance) = parent_edge.distance() {
		write!(w, ":{}{}", distance, parent_edge.attributes())?;
	}

	Ok(())
}
