use anyhow::Result;

use std::io::{BufReader, Cursor};

use data::nexus::{
	BlockReader, CommandReader, NexusTreeReader, TreeCommandRef,
};

fn commands(input: &str) -> Result<Vec<(String, usize, usize)>> {
	CommandReader::new(Cursor::new(input))?
		.map(|command| {
			let command = command?;
			Ok((
				command.source().to_owned(),
				command.line(),
				command.column(),
			))
		})
		.collect()
}

#[test]
fn reads_commands_and_locations() -> Result<()> {
	assert_eq!(
		commands("#nexus BEGIN TREES;\n  TREE one = (A,B); END;")?,
		[
			("BEGIN TREES;".to_owned(), 1, 8),
			("TREE one = (A,B);".to_owned(), 2, 3),
			("END;".to_owned(), 2, 21),
		]
	);
	Ok(())
}

#[test]
fn preserves_nested_comments_and_quoted_semicolons() -> Result<()> {
	let input = "#NEXUS\n[outer; [inner;] done] BEGIN TREES;\nTREE 'a;''b' = ('x;y',A[&note='z;']);\nEND;";
	let commands = commands(input)?;
	assert_eq!(commands.len(), 3);
	assert_eq!(commands[0].0, "[outer; [inner;] done] BEGIN TREES;");
	assert_eq!(commands[1].0, "TREE 'a;''b' = ('x;y',A[&note='z;']);");
	assert_eq!(commands[2].0, "END;");
	Ok(())
}

#[test]
fn accepts_bom_unicode_and_empty_files_after_header() -> Result<()> {
	assert!(commands("\u{feff}#NEXUS\n")?.is_empty());
	assert_eq!(
		commands("\u{feff}#NEXUS\nTITLE 'Árbol';")?[0].0,
		"TITLE 'Árbol';"
	);
	Ok(())
}

#[test]
fn ignores_comment_only_input() -> Result<()> {
	assert_eq!(
		commands("#NEXUS\n[before]; BEGIN NOTES; END; [after]")?,
		[
			("BEGIN NOTES;".to_owned(), 2, 11),
			("END;".to_owned(), 2, 24),
		]
	);
	Ok(())
}

#[test]
fn reads_across_small_buffers() -> Result<()> {
	let input = "#NEXUS\nBEGIN TREES;\nTREE one = (Á,'B;B');\nEND;";
	let expected = commands(input)?;
	for capacity in 1..=16 {
		let reader =
			BufReader::with_capacity(capacity, Cursor::new(input));
		let observed = CommandReader::new(reader)?
			.map(|command| {
				let command = command?;
				Ok((
					command.source().to_owned(),
					command.line(),
					command.column(),
				))
			})
			.collect::<Result<Vec<_>>>()?;
		assert_eq!(observed, expected);
	}
	Ok(())
}

#[test]
fn passes_borrowed_commands_from_reused_storage() -> Result<()> {
	let mut reader = CommandReader::new(Cursor::new(
		"#NEXUS\nTREE one = (A,B);\nTREE two = (C,D);",
	))?;
	let first = reader
		.next_command_with(|source, line, column| {
			assert_eq!((line, column), (2, 1));
			assert_eq!(source, "TREE one = (A,B);");
			Ok(source.as_ptr())
		})?
		.unwrap();
	let second = reader
		.next_command_with(|source, line, column| {
			assert_eq!((line, column), (3, 1));
			assert_eq!(source, "TREE two = (C,D);");
			Ok(source.as_ptr())
		})?
		.unwrap();
	assert_eq!(first, second);
	assert!(reader.next_command_with(|_, _, _| Ok(()))?.is_none());
	Ok(())
}

#[test]
fn passes_borrowed_tree_block_commands() -> Result<()> {
	let mut reader = BlockReader::new(Cursor::new(
		"#NEXUS\nBEGIN NOTES; TEXT ignored; END; BEGIN TREES; TREE a=(A,B); TREE b=(C,D); END;",
	))?;
	let mut seen = Vec::new();
	while reader
		.next_command_with(|block, name, source, _, _, _| {
			if block.eq_ignore_ascii_case("trees") {
				assert_eq!(name, "TREE");
				seen.push(source.to_owned());
			}
			Ok(())
		})?
		.is_some()
	{}
	assert_eq!(seen, ["TREE a=(A,B);", "TREE b=(C,D);"]);
	Ok(())
}

#[test]
fn borrows_tree_names_and_newick_text() -> Result<()> {
	let mut reader = BlockReader::new(Cursor::new(
		"#NEXUS\nBEGIN TREES; TREE sample = (A,'B; C'); END;",
	))?;
	let observed = reader.next_command_with(
		|block, name, source, _, line, column| {
			let tree = TreeCommandRef::parse(
				block, name, source, line, column,
			)?;
			assert_eq!(tree.name(), "sample");
			assert_eq!(tree.newick(), "(A,'B; C');");
			assert!(source
				.as_bytes()
				.as_ptr_range()
				.contains(&tree.name().as_ptr()));
			assert!(source
				.as_bytes()
				.as_ptr_range()
				.contains(&tree.newick().as_ptr()));
			Ok(())
		},
	)?;
	assert!(observed.is_some());
	Ok(())
}

#[test]
fn visits_tree_slices_with_translation_by_block() -> Result<()> {
	let input = "#NEXUS\nBEGIN NOTES; TEXT ignored; END; BEGIN TREES; TRANSLATE 1 A, 2 'B B'; TREE first=(1,'x;y'); TREE second=(1,2); END; BEGIN TREES; TREE third=(1,2); END;";
	let mut reader = NexusTreeReader::new(BufReader::with_capacity(
		3,
		Cursor::new(input),
	))?;
	let mut seen = Vec::new();
	reader.for_each_tree(|name, newick, translation| {
		seen.push((
			name.to_owned(),
			newick.to_owned(),
			translation
				.and_then(|table| table.get("2"))
				.map(str::to_owned),
		));
		Ok(())
	})?;
	assert_eq!(
		seen,
		[
			(
				"first".to_owned(),
				"(1,'x;y');".to_owned(),
				Some("B B".to_owned())
			),
			(
				"second".to_owned(),
				"(1,2);".to_owned(),
				Some("B B".to_owned())
			),
			("third".to_owned(), "(1,2);".to_owned(), None),
		]
	);
	Ok(())
}

#[test]
fn visits_many_trees_without_retaining_commands() -> Result<()> {
	let mut input = String::from("#NEXUS\nBEGIN TREES;");
	for _ in 0..10_000 {
		input.push_str("TREE sample=(A,B);\n");
	}
	input.push_str("END;");
	let mut reader = NexusTreeReader::new(Cursor::new(input))?;
	let mut count = 0;
	let mut pointer = None;
	reader.for_each_tree(|name, newick, translation| {
		assert_eq!(name, "sample");
		assert_eq!(newick, "(A,B);");
		assert!(translation.is_none());
		if let Some(previous) = pointer {
			assert_eq!(newick.as_ptr(), previous);
		} else {
			pointer = Some(newick.as_ptr());
		}
		count += 1;
		Ok(())
	})?;
	assert_eq!(count, 10_000);
	Ok(())
}

#[test]
fn stops_on_callback_error() -> Result<()> {
	let mut reader = NexusTreeReader::new(Cursor::new(
		"#NEXUS\nBEGIN TREES; TREE sample=(A,B); END;",
	))?;
	let error = reader.for_each_tree(|_, _, _| anyhow::bail!("stop here"));
	assert_eq!(error.unwrap_err().to_string(), "stop here");
	Ok(())
}

#[test]
fn rejects_invalid_input() {
	for (input, message) in [
		("BEGIN TREES;", "Expected '#NEXUS'"),
		("#NEXUSx\n", "Expected whitespace"),
		("#NEXUS\nTREE x = (A,B)", "Unterminated NEXUS command"),
		("#NEXUS\nTREE x = ('A,B);", "Unterminated quoted"),
		(
			"#NEXUS\nTREE x = (A[broken,B);",
			"Unterminated NEXUS comment",
		),
		("#NEXUS\nTREE x = (A],B);", "Unexpected ']'"),
	] {
		let error = commands(input).unwrap_err().to_string();
		assert!(
			error.contains(message),
			"expected {message:?} in {error:?}"
		);
	}
}
