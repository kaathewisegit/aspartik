import pytest

from aspartik.b3.parameters import Real, Weights


# pyright: reportUnusedExpression=false
def test_eq():
    assert Real(0.1) == 0.1
    assert Real(0.1) != Real(0.2)


def test_comparison_value():
    assert Real(1e-40) < 1
    assert Real(0) <= 0
    assert Real(2.0) > 1.0


def test_weights_comparable():
    assert Weights(0.1, 0.2, 0.3) < 0.4
    assert Weights(0.1, 0.2, 0.3) <= 0.3
    assert Weights(0.1, 0.2, 0.3) > 0.0
    assert Weights(0.1, 0.2, 0.3) >= 0.1

    assert not Weights(0.1, 0.2, 0.3) < 0.2
    assert not Weights(0.1, 0.2, 0.3) <= 0.2
    assert not Weights(0.1, 0.2, 0.3) > 0.2
    assert not Weights(0.1, 0.2, 0.3) >= 0.2
