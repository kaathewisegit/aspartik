use data::newick::{Edge, Node, Tree};

#[test]
fn serialize_basic() {
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

	let mut string = String::new();
	tree.serialize(&mut string).unwrap();

	assert_eq!(string, "(A:0.1,B,(C,D)E)F;");
}
