from cliclass import CliCommand

import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import Literal

from . import doc

FEATURES = "--features arbitrary"


@contextmanager
def chdir(dir: str):
    old_dir = os.getcwd()

    os.chdir(dir)

    try:
        yield
    finally:
        os.chdir(old_dir)


def execute(cmd: str, **kwargs):
    result = subprocess.run(cmd, shell=True, **kwargs)

    if result.returncode != 0:
        sys.exit(f"Command `{cmd}` failed with exit code {result.returncode}")

    return result


def is_ci():
    return os.environ.get("CI") is not None


def is_changed(dirs: list[str]):
    changed = execute("git status --porcelain", capture_output=True, text=True)
    changed = [line[3:] for line in changed.stdout.splitlines()]
    return any(any(line.startswith(dir) for dir in dirs) for line in changed)


def should_run(kind: Literal["rust", "python", "website"]):
    changed = False
    match kind:
        case "rust":
            changed = is_changed(["crates/", "Cargo"])
        case "python":
            changed = is_changed(["crates/", "Cargo", "python/", "pyproject"])
        case "website":
            changed = is_changed(["website/"])
    return is_ci() or changed


@dataclass(kw_only=True)
class Langopts:
    rust: bool = field(default=should_run("rust"))
    python: bool = field(default=should_run("python"))
    website: bool = field(default=should_run("website"))


@dataclass
class Fix(Langopts):
    """Automatically fix code"""

    def run(self):
        if self.rust:
            execute("cargo fmt")
            execute("cargo clippy --fix --allow-dirty")

        if self.python:
            execute("ruff format")
            execute("ruff check --extend-select F401 --fix")

        if self.website:
            with chdir("website/"):
                execute("npm run fix")


@dataclass
class Lint(Langopts):
    """Validate with linters and formatters"""

    def run(self):
        if self.rust:
            execute("cargo fmt --check")
            execute(f"cargo clippy --workspace --tests {FEATURES} -- -D warnings")

        if self.python:
            execute("ruff format --check")
            execute("ruff check --extend-select F401")
            execute("ty check")

        if self.website:
            with chdir("website/"):
                execute("npm run check")


@dataclass
class Test(Langopts):
    """Run tests"""

    def run(self):
        if self.rust and should_run("rust"):
            execute(f"cargo test --workspace {FEATURES}")

        if self.python and should_run("python"):
            execute("pytest")


@dataclass
class Run:
    """Run a minimal `b3` simulation"""

    def run(self):
        execute("uv run --no-sync python/examples/apes.py")
        influenza_len = 100 if is_ci() else 20_000
        execute(f"uv run --no-sync python/examples/influenza.py {influenza_len}")


@dataclass
class Check(Langopts):
    """Run all checks"""

    def run(self):
        Lint().run()
        Test().run()

        if self.rust or self.python:
            Run().run()


@dataclass
class Clean:
    """Remove temporary files"""

    target: bool = field(default=False, kw_only=True)
    "Cleanup Rust's `target/` dir too"

    def run(self):
        execute("ruff clean")

        for path in Path(".").glob("python/**/__pycache__/"):
            rmtree(path)

        for path in Path(".").glob("b3-error-*.state"):
            path.unlink()

        artifacts = [
            "flamegraph.svg",
            "perf.data",
            "perf.data.old",
            ".pytest_cache",
        ]
        for path in map(Path, artifacts):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)

        if self.target:
            execute("cargo clean")


@dataclass
class Build:
    """Build a wheel for the host platform"""

    def run(self):
        rmtree("target/wheels/", ignore_errors=True)
        execute("maturin build --release")


@dataclass
class Sdist:
    """Build sdist"""

    def run(self):
        rmtree("target/sdist/", ignore_errors=True)
        execute("maturin sdist --out target/sdist/")


@dataclass
class Pdoc:
    """Generate pdoc json dump"""

    def run(self):
        doc.dump_json("aspartik")


@dataclass
class Toolkit:
    """Aspartik development helper"""

    subcommand: Fix | Lint | Test | Run | Check | Clean | Build | Sdist | Pdoc = field(
        metadata={"cli": {"subcommand": True}}
    )

    def run(self):
        self.subcommand.run()


def main():
    cli = CliCommand(Toolkit)
    toolkit = cli.parse()
    toolkit.run()
