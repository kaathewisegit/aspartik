use anyhow::Result;

use std::io::Cursor;

use data::nexus::{NexusParser, read_binary_trees};

fn parse(source: &str) -> Result<()> {
	let mut p = NexusParser::default();
	for line in source.lines() {
		p.parse_line(line, |_, _| Ok(()))?;
	}
	p.finish()?;
	Ok(())
}

#[test]
fn valid_files() {
	let sources = [
		r"#NEXUS
BEGIN TAXA;
	TREE tree = (X:1,Y:1);
END;
BEGIN TREES;
	TREE first = [&R](A:1,B:2);
	TREE second [&U] =
		(A:3,B:4);
END;",
		r"#NEXUS
BEGIN TREES;
	TREE tree = (A[semicolon;inside;a;comment]:1,C:2);
END;",
		r"#NEXUS
	[before
	[nested;]
	after] BEGIN TREES;
TREE [name
	[part;]
	] tree = (A:1,B:2);
END;",
	];
	for source in sources {
		parse(source).unwrap();
	}
}

#[test]
fn invalid_files() {
	let sources = [
		r"#NEXUS
BEGIN trees;
	tree x = (a:1,b:2)",
		r"#NEXUS
BEGIN trees;
	tree x = (a:1,b:2);",
		r"#NEXUS
BEGIN trees;
	[unreminated tree x = (a:1,b:2);
end;",
	];
	for source in sources {
		parse(source).unwrap_err();
	}
}

#[test]
fn trees() {
	let s = r"#NEXUS

BEGIN TAXA;
	DIMENSIONS ntax=4;
	TAXLABELS
		Taxon1
		Taxon2
		Taxon3
		Taxon4
	;
END;

BEGIN TREES;
	TRANSLATE
		1 Taxon1,
		2 Taxon2,
		3 Taxon3,
		4 Taxon4;
	TREE tree1 = [&R] (((1:0.1, 2:0.2):0.3, 3:0.4):0.5, 4:0.6);
	TREE tree2 = [&R] ((1:0.2, (2:0.1, 3:0.3):0.2):0.4, 4:0.5);
END;
	";

	let expected_trees = [
		"(((Taxon1:0.1,Taxon2:0.2):0.3,Taxon3:0.4):0.5,Taxon4:0.6);",
		"((Taxon1:0.2,(Taxon2:0.1,Taxon3:0.3):0.2):0.4,Taxon4:0.5);",
	];

	let trees = read_binary_trees(Cursor::new(s)).unwrap();

	for (tree, expected) in trees.iter().zip(expected_trees) {
		assert_eq!(tree.to_newick().unwrap(), expected);
	}
}

#[test]
fn trees_quoted_translation_names() {
	let s = r#"#NEXUS

BEGIN TAXA;
	DIMENSIONS ntax=3;
	TAXLABELS "Taxon one" 'Taxon [two]' Taxon3;
END;

BEGIN TREES;
	TRANSLATE
		1 "Taxon one",
		2 'Taxon [two]',
		3 Taxon3;
	TREE tree1 = [&R] ((1:0.1, 2:0.2):0.3, 3:0.4);
	TREE tree2 = [&R] (1:0.2, (2:0.1, 3:0.3):0.2);
END;
	"#;

	let expected_trees = [
		"(('Taxon one':0.1,'Taxon [two]':0.2):0.3,Taxon3:0.4);",
		"('Taxon one':0.2,('Taxon [two]':0.1,Taxon3:0.3):0.2);",
	];

	let trees = read_binary_trees(Cursor::new(s)).unwrap();

	for (tree, expected) in trees.iter().zip(expected_trees) {
		assert_eq!(tree.to_newick().unwrap(), expected);
	}
}
