# Contributing

## Setup

To build Aspartik you'll need a recent version Rust/Cargo (1.98.0 as of the time of writing)[^rt] and uv ([installation instructions][uv]).

After cloning the repository run `uv sync` to setup everything.  This might take awhile, as it'll build the entire Rust workspace.

`uv run -m python.toolkit` provides a number of utilities: linting, running tests, etc.

To activate the uv virtual environment run `source .venv/bin/activate` on Linux/macOS and `.venv\Scripts\activate` on Windows[^venv].

There are several example configurations at `python/examples` which I use as smoke tests.  All other checks (linting, formatting, tests) can be ran with `uv -m python.toolkit check`.  Both tests and examples use data from another repository in a submodule.  Run `git submodule update --init` to fetch it.


## Licensing

Before making a contribution, please read [the CLA][cla] (it's short).  It gives me the right to relicense the project or its parts (including your contributions) under other copyleft licenses or MPL-2.0.  If You agree to the terms, commit with the `--signoff` or add a `Signed-off-by:` trailer to the PR description.


[^rt]: I'm not using `rust-toolchain.toml` because it's not forward compatible with never versions.

[^venv]: See the docs for other supported shells here: [Using Python environments | uv][uv-venv].

[uv]: https://docs.astral.sh/uv/getting-started/installation/
[uv-venv]: https://docs.astral.sh/uv/pip/environments/
[cla]: docs/CLA.txt
