import pytest
from utils import random_float, random_frequencies_many

from aspartik.b3.substitutions import F81, GTR, HKY, Matrix
from aspartik.math import is_close


def assert_normalized(frequencies, matrix: Matrix) -> None:
    total = 0
    for i in range(4):
        total += frequencies[i] * matrix[i][i]

    assert is_close(total, 1.0)


@pytest.mark.parametrize("frequencies", random_frequencies_many())
def test_f81_normalization(frequencies):
    model = F81(frequencies)
    assert_normalized(frequencies, model.get_matrix())


@pytest.mark.parametrize(
    "frequencies,kappa", zip(random_frequencies_many(), random_float(0, 10, num=1000))
)
def test_hky_normalization(frequencies, kappa: float):
    model = HKY(frequencies, kappa)
    assert_normalized(frequencies, model.get_matrix())


@pytest.mark.parametrize(
    "frequencies,rate_ac,rate_ag,rate_at,rate_cg,rate_ct,rate_gt",
    zip(
        random_frequencies_many(num=2000),
        random_float(0, 10, num=2000),
        random_float(0, 10, num=2000),
        random_float(0, 10, num=2000),
        random_float(0, 10, num=2000),
        random_float(0, 10, num=2000),
        random_float(0, 10, num=2000),
    ),
)
def test_gtr_normalization(
    frequencies,
    rate_ac: float,
    rate_ag: float,
    rate_at: float,
    rate_cg: float,
    rate_ct: float,
    rate_gt: float,
):
    model = GTR(frequencies, rate_ac, rate_ag, rate_at, rate_cg, rate_ct, rate_gt)
    assert_normalized(frequencies, model.get_matrix())
