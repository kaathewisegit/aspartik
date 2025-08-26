from argparse import ArgumentParser
from shutil import rmtree
import subprocess
import sys
from pathlib import Path


def make_parser():
    parser = ArgumentParser(
        prog="toolkit",
        description="Command runner for working on Aspartik",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    subparsers.add_parser("lint", help="validate with linters and formatters")
    subparsers.add_parser("test", help="run tests")
    subparsers.add_parser("run", help="run a minimal `b3` simulation")
    check = subparsers.add_parser("check", help="run all checks")  # noqa: F841
    subparsers.add_parser("clean", help="remove temporary files and `b3` output")

    subparsers.add_parser("pdoc", help="build the pdoc HTML files")

    return parser


def execute(*args):
    result = subprocess.run(args)

    if result.returncode != 0:
        sys.exit(
            f"Command `{' '.join(args)}` failed with exit code {result.returncode}"
        )


def lint():
    execute("cargo", "fmt", "--check")
    execute(
        "cargo",
        "clippy",
        "--workspace",
        "--tests",
        "--features",
        "approx,arbitrary",
        "--",
        "-D",
        "warnings",
    )

    execute("ruff", "format", "--check")
    execute("ruff", "check")

    execute("pyright")


def test():
    execute("cargo", "test", "--workspace", "--features", "approx,arbitrary")

    execute("pytest")


def run():
    execute("maturin", "develop", "--release")
    execute("python", "python/benches/primate.py")


ARTIFACTS = [
    "flamegraph.svg",
    "perf.data",
    "perf.data.old",
    "b3.trees",
    "b3.log",
    ".pytest_cache",
]


def check():
    lint()
    test()
    run()


def clean():
    execute("ruff", "clean")

    for path in Path(".").glob("python/**/__pycache__/"):
        rmtree(path)

    for path in map(Path, ARTIFACTS):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


def pdoc():
    execute(
        "pdoc", "--no-search", "-t", "docs/template", "-o", "target/pdoc/", "aspartik"
    )


def main():
    parser = make_parser()
    args = parser.parse_args()

    match args.subcommand:
        case "lint":
            lint()
        case "test":
            test()
        case "run":
            run()
        case "check":
            check()
        case "clean":
            clean()
        case "pdoc":
            pdoc()
        case None:
            parser.print_help()
