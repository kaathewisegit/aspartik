def "metadata target" [] {
	cargo metadata --format-version 1 | from json | get target_directory
}

def "metadata root" [] {
	cargo metadata  --format-version 1 | from json | get workspace_root
}

# Run all checks on the repository
export def check [] {
	ruff format --check
	ruff check

	pyright

	cargo clippy --workspace -- -D warnings
	cargo test --workspace --features approx,proptest
}

# Remove temporary files and `b3` output
export def clean [] {
	ruff clean
	(
		rm --permanent --force --recursive
			**/flamegraph.svg
			**/perf.data **/perf.data.old
			crates/**/__pycache__/
			**/b3.trees
	)
}
