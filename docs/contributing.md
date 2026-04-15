# Contributing

## Setup

To build Aspartik you'll need a recent version Rust/Cargo (1.91 as of
the time of writing)[^rt] and uv ([installation instructions][uv]).

After cloning the repository run `uv sync` to setup everything.  This
might take awhile, as it'll build the Rust package.

`uv -m python.toolkit` provides a number of utilities: linting, running
tests, etc.

To activate the uv virtual environment run `source .venv/bin/activate`
on Linux/macOS and `.venv\Scripts\activate` on Windows[^venv].

There are currently 3 example configurations at `python/examples`.  All
tests can be ran with `uv -m python.toolkit test`.  Both tests and
examples use data from another repository in a submodule.  Run `git
submodule update --init` to fetch it.


[^rt]: I'm not using `rust-toolchain.toml` because it's not forward
    compatible with never versions.  In the future I'll add
    `rust-version` to `Cargo.toml`.

[^venv]: See the docs for other supported shells here: [Using Python
    environments | uv][uv-venv].

[uv]: https://docs.astral.sh/uv/getting-started/installation/
[uv-venv]: https://docs.astral.sh/uv/pip/environments/
