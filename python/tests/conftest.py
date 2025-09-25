from pytest import Config, TestReport, fixture

from aspartik.rng import RNG

_rng = RNG(4)


def pytest_report_teststatus(report: TestReport, config: Config):
    if report.outcome != "passed" or report.when != "call":
        return None  # handled by pytest

    letter = ""
    if _rng.random_bool(0.01):
        letter = "."
    return report.outcome, letter, report.outcome.upper()


@fixture
def rng():
    return RNG(4)
