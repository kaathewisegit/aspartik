use std::fmt::{Result, Write};

use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};

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
	let root = tree.root.unwrap_or_else(|| {
		// pick a random node if there's no root
		tree.graph.node_identifiers().next().unwrap()
	});

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

		let edges: Vec<_> = tree.edges_of(node).collect();

		if edges.is_empty() {
			// leaf
			write_node(tree.get_node(node), edge, writer)?;
			write_comma!();
		} else {
			writer.write_char('(')?;

			stack.push((node, edge, true));

			for edge in edges {
				stack.push((
					edge.target(),
					edge.weight(),
					false,
				));
			}
		}
	}

	writer.write_char(';')
}

fn write_node<W: Write>(
	node: &Node,
	parent_edge: &Edge,
	writer: &mut W,
) -> Result {
	if !node.name().is_empty() {
		write!(writer, "{}{}", node.name(), node.attributes())?;
	}
	if let Some(distance) = parent_edge.distance() {
		write!(writer, ":{}{}", distance, parent_edge.attributes())?;
	}

	Ok(())
}

// TODO: move to integration tests
#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn basic() {
		let mut tree = Tree::new();

		const EMPTY: String = String::new();

		let a = tree.add_node(Node::new("A".to_owned(), EMPTY));
		let b = tree.add_node(Node::new("B".to_owned(), EMPTY));
		let c = tree.add_node(Node::new("C".to_owned(), EMPTY));
		let d = tree.add_node(Node::new("D".to_owned(), EMPTY));
		let e = tree.add_node(Node::new("E".to_owned(), EMPTY));
		let f = tree.add_node(Node::new("F".to_owned(), EMPTY));

		tree.add_edge(f, a, Edge::new(Some(0.1), EMPTY));
		tree.add_edge(f, b, Edge::new(None, EMPTY));
		tree.add_edge(f, e, Edge::new(None, EMPTY));
		tree.add_edge(e, c, Edge::new(None, EMPTY));
		tree.add_edge(e, d, Edge::new(None, EMPTY));

		tree.set_root(f);

		let mut newick = String::new();
		serialize(&tree, &mut newick).unwrap();

		assert_eq!(newick, "(A:0.1,B,(C,D)E)F;");
	}
}
