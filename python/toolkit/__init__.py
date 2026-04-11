from clypi import ClypiConfig, ClypiFormatter, Command, arg, configure
from typing_extensions import override

import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from typing import Literal

from . import doc


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


class Fix(Command):
    """Automatically fix code"""

    rust: bool = arg(inherited=True)
    python: bool = arg(inherited=True)
    website: bool = arg(inherited=True)

    @override
    async def run(self):
        if self.rust:
            execute("cargo fmt")
            execute("cargo clippy --fix --allow-dirty")

        if self.python:
            execute("ruff format")
            execute("ruff check --extend-select F401 --fix")

        if self.website:
            with chdir("website/"):
                execute("npm run fix")


class Lint(Command):
    """Validate with linters and formatters"""

    rust: bool = arg(inherited=True)
    python: bool = arg(inherited=True)
    website: bool = arg(inherited=True)

    @override
    async def run(self):
        if self.rust:
            execute("cargo fmt --check")
            execute(
                "cargo clippy --workspace --tests --features arbitrary -- -D warnings"
            )

        if self.python:
            execute("ruff format --check")
            execute("ruff check --extend-select F401")
            execute("ty check")

        if self.website:
            with chdir("website/"):
                execute("npm run check")


class Test(Command):
    """Run tests"""

    rust: bool = arg(inherited=True)
    python: bool = arg(inherited=True)

    @override
    async def run(self):
        if self.rust and should_run("rust"):
            execute("cargo test --workspace --features arbitrary")

        if self.python and should_run("python"):
            execute("pytest")


class Run(Command):
    """Run a minimal `b3` simulation"""

    @override
    async def run(self):
        execute("uv run --no-sync python/examples/apes.py")
        influenza_len = 100 if is_ci() else 20_000
        execute(f"uv run --no-sync python/examples/influenza.py {influenza_len}")


class Check(Command):
    """Run all checks"""

    rust: bool = arg(inherited=True)
    python: bool = arg(inherited=True)
    website: bool = arg(inherited=True)

    @override
    async def run(self):
        await Lint(self.rust, self.python, self.website).run()
        await Test(self.rust, self.python).run()

        if self.rust or self.python:
            await Run().run()


class Clean(Command):
    """Remove temporary files"""

    target: bool = arg(False, help="Cleanup Rust's `target/` dir too")

    @override
    async def run(self):
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


class Build(Command):
    """Build a wheel for the host platform"""

    @override
    async def run(self):
        rmtree("target/wheels/", ignore_errors=True)
        execute("maturin build --release")

        if platform.system() == "Windows":
            wheel_dir = Path("target/wheels/")
            wheel_path = next(wheel_dir.iterdir())
            paths = "C:/mingw64/bin/;C:/msys64/ucrt64/bin/;C:/msys64/mingw64/bin/"
            execute(
                f"delvewheel repair --add-path {paths} --include libgfortran-5.dll {wheel_path}"
            )
            execute(f"uv pip install wheelhouse/{wheel_path.name}")


class Sdist(Command):
    """Build sdist"""

    @override
    async def run(self):
        rmtree("target/sdist/", ignore_errors=True)
        execute("maturin sdist --out target/sdist/")


class Pdoc(Command):
    """Generate pdoc json dump"""

    @override
    async def run(self):
        doc.dump_json("aspartik")


class Toolkit(Command):
    """Aspartik development helper"""

    subcommand: Fix | Lint | Test | Run | Check | Clean | Build | Sdist | Pdoc | None

    rust: bool = arg(default=should_run("rust"))
    python: bool = arg(default=should_run("python"))
    website: bool = arg(default=should_run("website"))

    @override
    async def run(self):
        self.print_help()


def main():
    configure(
        ClypiConfig(help_formatter=ClypiFormatter(boxed=False, show_option_types=False))
    )

    cli = Toolkit.parse()
    cli.start()
