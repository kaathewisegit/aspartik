def "metadata target" [] {
	cargo metadata --format-version 1 | from json | get target_directory
}

def "metadata root" [] {
	cargo metadata  --format-version 1 | from json | get workspace_root
}

export def lint [] {
	ruff format --check
	ruff check

	pyright

	cargo fmt --check
	cargo clippy --workspace -- -D warnings
}

export def test [
	--no-rust
	--no-python
] {
	if not $no_rust {
		cargo test --workspace --features approx,proptest
	}
	if not $no_python {
		pytest
	}
}

export def run [] {
	maturin develop --release
	timeit { python3 benches/primate.py }
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
			tracing.log.*
			crates/**/__pycache__/
			.pytest_cache/
	)
}
