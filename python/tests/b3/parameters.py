from aspartik.b3.parameters import IntVector, Real, RealVector


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


def test_int_vector_indexing():
    vec = IntVector(10, 20, 30)
    assert vec[0] == 10
    assert vec[2] == 30
    vec[1] = 25
    assert vec[1] == 25
    assert len(vec) == 3


def test_int_vector_comparable():
    vec = IntVector(1, 2, 3)
    assert vec < 4
    assert vec <= 3
    assert vec > 0
    assert vec >= 1
    assert vec != IntVector(3, 2, 1)

    assert not vec < 2
    assert not vec <= 1
    assert not vec > 3
    assert not vec >= 4


def test_int_vector_negative_values():
    vec = IntVector(-10, 0, 10)
    assert vec[0] == -10
    assert vec[1] == 0

    assert not vec < 0
    assert vec >= -10
    assert vec > -30
    assert vec >= -20
    assert not vec < -25
    assert not vec > 15

    vec[2] = -20
    assert vec[2] == -20
