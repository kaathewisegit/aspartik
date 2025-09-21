from pytest import Config, TestReport

from aspartik.rng import RNG

_rng = RNG(4)


def pytest_report_teststatus(report: TestReport, config: Config):
    if report.outcome != "passed" or report.when != "call":
        return None  # handled by pytest

    letter = ""
    if _rng.random_bool(0.01):
        letter = "."
    return report.outcome, letter, report.outcome.upper()


def random_float(lower: float, upper: float, num: int = 100) -> list[float]:
    rng = RNG(4)
    return [rng.random_float() for _ in range(num)]


def random_integer(lower: int, upper: int, num: int = 100) -> list[int]:
    rng = RNG(4)
    return [rng.random_int(lower, upper) for _ in range(num)]
