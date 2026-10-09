use anyhow::{Result, bail};

use std::io::Cursor;

use data::nexus::for_each_tree;

#[test]
fn reads_beast_trees() -> Result<()> {
	let source = concat!(
		"#NEXUS\n",
		"Begin taxa;\n",
		"Dimensions ntax=3;\n",
		"End;\n",
		"Begin trees;\n",
		"Translate\n",
		" 2 B,\n",
		" 3 'C C',\n",
		" 1 A;\n",
		"tree STATE_0 = [&R] ((1[&date=2020]:0.1,2:0.2):0.3,3:0.4);\n",
		"tree STATE_10000 = [&R] (1:0.2,(2:0.3,3:0.4):0.5);\n",
		"End;\n",
	);
	let mut trees = Vec::new();
	for_each_tree(Cursor::new(source), |tree, translation| {
		let (aliases, taxa) = translation.unwrap();
		assert_eq!(aliases.iter().collect::<Vec<_>>(), ["1", "2", "3"]);
		assert_eq!(taxa.iter().collect::<Vec<_>>(), ["A", "B", "C C"]);
		trees.push(tree.to_owned());
		Ok(())
	})?;
	assert_eq!(
		trees,
		[
			"[&R] ((1[&date=2020]:0.1,2:0.2):0.3,3:0.4);",
			"[&R] (1:0.2,(2:0.3,3:0.4):0.5);",
		]
	);
	Ok(())
}

#[test]
fn reads_mrbayes_trees() -> Result<()> {
	let source = concat!(
		"#NEXUS\n",
		"[ID: 0184438524]\n",
		"[Param: tree]\n",
		"begin trees;\n",
		"translate\n",
		"1 Tarsius_syrichta,\n",
		"2 'Lemur catta',\n",
		"3 'O''Brien';\n",
		"tree gen.0 = [&U] ((1:2.0e-02,2:2.0e-02):1.0e-01,3:2.0e-01);\n",
		"end;\n",
	);
	let mut count = 0;
	for_each_tree(Cursor::new(source), |tree, translation| {
		let (aliases, taxa) = translation.unwrap();
		assert_eq!(aliases.iter().collect::<Vec<_>>(), ["2", "3", "1"]);
		assert_eq!(
			taxa.iter().collect::<Vec<_>>(),
			["Lemur catta", "O'Brien", "Tarsius_syrichta"]
		);
		assert_eq!(
			tree,
			"[&U] ((1:2.0e-02,2:2.0e-02):1.0e-01,3:2.0e-01);"
		);
		count += 1;
		Ok(())
	})?;
	assert_eq!(count, 1);
	Ok(())
}

#[test]
fn reads_beast_translation_terminated_on_the_next_line() -> Result<()> {
	let source = "#NEXUS\nBegin trees;\nTranslate\n1 A,\n2 B\n;\ntree STATE_0 = (1:0.1,2:0.2);\nEnd;\n";
	let mut count = 0;
	for_each_tree(Cursor::new(source), |tree, translation| {
		let (aliases, taxa) = translation.unwrap();
		assert_eq!(aliases.iter().collect::<Vec<_>>(), ["1", "2"]);
		assert_eq!(taxa.iter().collect::<Vec<_>>(), ["A", "B"]);
		assert_eq!(tree, "(1:0.1,2:0.2);");
		count += 1;
		Ok(())
	})?;
	assert_eq!(count, 1);
	Ok(())
}

#[test]
fn ignores_other_blocks_and_resets_translation() -> Result<()> {
	let source = concat!(
		"#NEXUS\n",
		"BEGIN TAXA;\n",
		"TREE ignored = (X:1,Y:1);\n",
		"END;\n",
		"BEGIN TREES;\n",
		"TRANSLATE 2 B,\n",
		"1 A;\n",
		"TREE first = (1:1,2:2);\n",
		"END;\n",
		"BEGIN TREES;\n",
		"TREE second = (A:1,B:2);\n",
		"END;\n",
	);
	let mut trees = Vec::new();
	for_each_tree(Cursor::new(source), |tree, translation| {
		trees.push((tree.to_owned(), translation.is_some()));
		Ok(())
	})?;
	assert_eq!(
		trees,
		[
			("(1:1,2:2);".to_owned(), true),
			("(A:1,B:2);".to_owned(), false)
		]
	);
	Ok(())
}

#[test]
fn rejects_invalid_translation() {
	for table in [
		"1 A,\n1 B;",
		"1 A,\n2 A;",
		"1 A,\n2;",
		"1 A,\n2 B,",
		"1 A\n2 B;",
		"1[inline] A;",
		"1 A[inline];",
	] {
		let source = format!(
			"#NEXUS\nBEGIN TREES;\nTRANSLATE\n{table}\nEND;\n"
		);
		assert!(for_each_tree(Cursor::new(source), |_, _| Ok(()))
			.is_err());
	}
}

#[test]
fn rejects_incomplete_trees() {
	for tree in [
		"TREE x = (A:1,B:2)",
		"TREE = (A:1,B:2);",
		"TREE x (A:1,B:2);",
	] {
		let source = format!("#NEXUS\nBEGIN TREES;\n{tree}\nEND;\n");
		assert!(for_each_tree(Cursor::new(source), |_, _| Ok(()))
			.is_err());
	}
	assert!(for_each_tree(
		Cursor::new("#NEXUS\nBEGIN TREES;\n"),
		|_, _| Ok(())
	)
	.is_err());
}

#[test]
fn forwards_callback_error() {
	let source = "#NEXUS\nBEGIN TREES;\nTREE x = (A:1,B:2);\nEND;\n";
	let error = for_each_tree(Cursor::new(source), |_, _| {
		bail!("callback failed")
	})
	.unwrap_err();
	assert_eq!(error.to_string(), "callback failed");
}
