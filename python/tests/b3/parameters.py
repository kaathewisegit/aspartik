from aspartik.b3.parameters import Real, RealVector


# pyright: reportUnusedExpression=false
def test_eq():
    assert Real(0.1) == 0.1
    assert Real(0.1) != Real(0.2)


def test_comparison_value():
    assert Real(1e-40) < 1
    assert Real(0) <= 0
    assert Real(2.0) > 1.0


def test_weights_comparable():
    assert RealVector(0.1, 0.2, 0.3) < 0.4
    assert RealVector(0.1, 0.2, 0.3) <= 0.3
    assert RealVector(0.1, 0.2, 0.3) > 0.0
    assert RealVector(0.1, 0.2, 0.3) >= 0.1

    assert not RealVector(0.1, 0.2, 0.3) < 0.2
    assert not RealVector(0.1, 0.2, 0.3) <= 0.2
    assert not RealVector(0.1, 0.2, 0.3) > 0.2
    assert not RealVector(0.1, 0.2, 0.3) >= 0.2
