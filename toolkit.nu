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

export def test [] {
	cargo test --workspace --features approx,proptest
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
			tracing.log.*
			crates/**/__pycache__/
	)
}
