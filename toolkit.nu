def target_dir [] {
	cargo metadata --format-version 1 | from json | get target_directory
}

export def flame [
	package: string
	bench: string
	name?: string
] {
	let dst = target_dir | path join "flamegraph/likelihood.svg"
	mkdir ($dst | path dirname)

	(
		cargo flamegraph
			--package $package
			--bench $bench
			--output $dst
			--palette rust
			--skip-after "criterion::bencher::Bencher<M>::iter"
			--
			$name
	)

	rm --permanent --force **/perf.data*
}

# Run all checks on the repository
export def check [] {
	try {
		ruff format --check
		ruff check
	} catch {
		error make --unspanned {msg: "Python linting failed"}
	}

	try {
		cargo clippy --workspace -- -D warnings
	} catch {
		error make --unspanned {msg: "Rust linting failed"}
	}

	try {
		cargo test --workspace
	} catch {
		error make --unspanned {msg: "Rust tests failed"}
	}
}

# Remove temporary files
export def clean [] {
	ruff clean
	(
		rm --permanent --force --recursive
			**/flamegraph.svg
			**/perf.data **/perf.data.old
			crates/**/__pycache__
			crates/b3/b3.trees
	)
}

export def develop [] {
	let root = cargo metadata --format-version 1
		| from json
		| get workspace_root

	cd $root
	cargo build --release --workspace

	cp target/release/libstats.so crates/stats/stats/_stats_rust_impl.so
	cp target/release/libb3.so crates/b3/b3/_b3_rust_impl.so

	{
		cd crates/rng/
		uv pip install rng/
	}
	{
		cd crates/stats/
		uv pip install stats/
	}
	{
		cd crates/b3/
		uv pip install b3/
	}

	return
}
