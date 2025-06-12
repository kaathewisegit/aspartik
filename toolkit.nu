def "metadata target" [] {
	cargo metadata --format-version 1 | from json | get target_directory
}

def "metadata root" [] {
	cargo metadata  --format-version 1 | from json | get workspace_root
}

# Validate with linters and type checkers
export def lint [
	--rust
	--python
] {
	let all = $rust == false and $python == false
	let $rust = $rust or $all
	let $python = $python or $all

	if $rust {
		cargo fmt --check
		cargo clippy --workspace -- -D warnings
	}

	if $python {
		ruff format --check
		ruff check

		pyright
	}
}

# Run all tests
export def test [
	--rust
	--python
] {
	let all = $rust == false and $python == false
	let $rust = $rust or $all
	let $python = $python or $all

	if $rust {
		cargo test --workspace --features approx,proptest
	}
	if $python {
		pytest
	}
}

# Runs a smoke test
export def run [] {
	maturin develop --release
	timeit { python3 benches/primate.py }
}

# Run all checks
export def check [
	--rust
	--python
] {
	let all = $rust == false and $python == false
	let $rust = $rust or $all
	let $python = $python or $all

	lint --rust=$rust --python=$python
	test --rust=$rust --python=$python
	run
}

# Remove temporary files and `b3` output
export def clean [] {
	ruff clean
	(
		rm --permanent --force --recursive
			flamegraph.svg
			perf.data perf.data.old
			b3.trees
			b3.log
			aspartik-*.log
			crates/**/__pycache__/
			.pytest_cache/
	)
}
