import os
import platform
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

from . import doc


def add_langopts(parser: ArgumentParser):
    parser.add_argument("--rust", action="store_true", help="run Rust commands")
    parser.add_argument("--python", action="store_true", help="run Python commands")
    parser.add_argument(
        "--website", action="store_true", help="run JavaScript commands"
    )


def set_lang_args(args: Namespace):
    if "rust" not in args:
        return

    if not args.rust and not args.python and not args.website:
        args.rust = True
        args.python = True
        args.website = True


def make_parser():
    parser = ArgumentParser(
        prog="toolkit",
        description="Command runner for working on Aspartik",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    fix = subparsers.add_parser("fix", help="automatically fix code")
    add_langopts(fix)

    lint = subparsers.add_parser("lint", help="validate with linters and formatters")
    add_langopts(lint)

    test = subparsers.add_parser("test", help="run tests")
    add_langopts(test)

    subparsers.add_parser("run", help="run a minimal `b3` simulation")

    check = subparsers.add_parser("check", help="run all checks")
    add_langopts(check)

    subparsers.add_parser("clean", help="remove temporary files and `b3` output")

    subparsers.add_parser("pdoc", help="build the pdoc HTML files")

    subparsers.add_parser("build", help="build the platform wheel")

    subparsers.add_parser("sdist", help="build sdist")

    return parser


@contextmanager
def chdir(dir: str):
    old_dir = os.getcwd()

    os.chdir(dir)

    try:
        yield
    finally:
        os.chdir(old_dir)


def execute(cmd: str):
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        sys.exit(f"Command `{cmd}` failed with exit code {result.returncode}")


def fix(args: Namespace):
    if args.rust:
        execute("cargo fmt")
        execute("cargo clippy --fix --allow-dirty")

    if args.python:
        execute("ruff format")
        execute("ruff check --fix")

    if args.website:
        with chdir("website/"):
            execute("npm run fix")


def lint(args: Namespace):
    if args.rust:
        execute("cargo fmt --check")
        execute(
            "cargo clippy --workspace --tests --features arbitrary -- -D warnings",
        )

    if args.python:
        execute("ruff format --check")
        execute("ruff check")

        execute("ty check")

    if args.website:
        with chdir("website/"):
            execute("npm run check")


def test(args: Namespace):
    if args.rust:
        execute("cargo test --workspace --features arbitrary")

    if args.python:
        execute("pytest")


def run():
    execute("uv run --no-sync python/examples/apes.py")
    execute("uv run --no-sync python/examples/influenza.py 20_000")


ARTIFACTS = [
    "flamegraph.svg",
    "perf.data",
    "perf.data.old",
    "b3.trees",
    "b3.log",
    ".pytest_cache",
]


def check(args: Namespace):
    lint(args)
    test(args)
    run()


def clean():
    execute("ruff clean")

    for path in Path(".").glob("python/**/__pycache__/"):
        rmtree(path)

    for path in map(Path, ARTIFACTS):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


def build(args: Namespace):
    rmtree("target/wheels/", ignore_errors=True)

    if platform.system() == "Linux" and platform.libc_ver()[0] == "glibc":
        # support glibc 2.17
        execute("maturin build --release --compatibility manylinux2014")
    else:
        execute("maturin build --release")

    if platform.system() == "Windows":
        wheel_dir = Path("target/wheels/")
        wheel_path = next(wheel_dir.iterdir())

        paths = "C:/mingw64/bin/;C:/msys64/ucrt64/bin/;C:/msys64/mingw64/bin/"
        execute(
            f"delvewheel repair --add-path {paths} --include libgfortran-5.dll {wheel_path}"
        )
        execute(f"uv pip install wheelhouse/{wheel_path.name}")


def sdist():
    rmtree("target/sdist/", ignore_errors=True)
    execute("maturin sdist --out target/sdist/")


def pdoc():
    doc.dump_json("aspartik")


def main():
    parser = make_parser()
    args = parser.parse_args()
    set_lang_args(args)

    match args.subcommand:
        case "fix":
            fix(args)
        case "lint":
            lint(args)
        case "test":
            test(args)
        case "run":
            run()
        case "check":
            check(args)
        case "clean":
            clean()
        case "pdoc":
            pdoc()
        case "build":
            build(args)
        case "sdist":
            sdist()
        case None:
            parser.print_help()
