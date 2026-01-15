import pytest
from pytest import Config, TestReport, fixture

from aspartik.rng import RNG

_rng = RNG(4)


def pytest_report_teststatus(report: TestReport, config: Config):
    global _rng

    if report.outcome != "passed" or report.when != "call":
        return None  # handled by pytest

    letter = ""
    if _rng.random_bool(0.01):
        letter = "."
    return report.outcome, letter, report.outcome.upper()


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m") == "manual":
        return

    skip_manual = pytest.mark.skip(reason="need -m manual option to run")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(skip_manual)


@fixture
def rng():
    return RNG(4)
