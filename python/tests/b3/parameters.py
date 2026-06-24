import pytest

from math import inf

from aspartik.b3.parameters import ClassVector, IntVector, Real, RealVector


def test_real_comparisons():
    val = Real(0.2)

    assert val == 0.2
    assert val == Real(0.2)
    assert val != 0.1
    assert val != Real(0.1)

    assert val < 0.3
    assert val <= 0.2
    assert val > 0.1
    assert val >= 0.2

    assert not val < 0.1
    assert not val <= 0.1
    assert not val > 0.3
    assert not val >= 0.3


def test_real_vector_repeat():
    for length in range(100):
        vec = RealVector.repeat(0.5, length)
        for i in range(length):
            assert vec[i] == 0.5


def test_real_vector_bound():
    vec = RealVector(0.1, 0.2, 0.3)

    assert vec.is_bound(-inf, inf)

    assert vec.is_bound(0.0, inf)
    assert vec.is_bound(0.1, inf)
    assert not vec.is_bound(0.2, inf)

    assert vec.is_bound(-inf, 0.4)
    assert not vec.is_bound(-inf, 0.3)
    assert not vec.is_bound(-inf, 0.2)

    assert vec.is_bound(0.1, 0.4)
    assert not vec.is_bound(0.1, 0.3)
    assert not vec.is_bound(0.2, 0.4)
    assert not vec.is_bound(0.1, 0.2)


def test_int_vector_repeat():
    for length in range(100):
        vec = IntVector.repeat(10, length)
        for i in range(length):
            assert vec[i] == 10


def test_int_vector_indexing():
    vec = IntVector(10, 20, 30)
    assert vec[0] == 10
    assert vec[2] == 30
    vec[1] = 25
    assert vec[1] == 25
    assert len(vec) == 3


def test_int_vector_bound():
    vec = IntVector(1, 2, 3)

    assert vec.is_bound(None, None)

    assert vec.is_bound(0, None)
    assert vec.is_bound(1, None)
    assert not vec.is_bound(2, None)

    assert vec.is_bound(None, 4)
    assert not vec.is_bound(None, 3)
    assert not vec.is_bound(None, 2)

    assert vec.is_bound(1, 4)
    assert not vec.is_bound(1, 3)
    assert not vec.is_bound(2, 4)
    assert not vec.is_bound(1, 2)


def test_classvec_constraints():
    with pytest.raises(match="at least 2 classes"):
        ClassVector(0, 1)
        ClassVector(1, 100)

    with pytest.raises(match="non-empty"):
        ClassVector(0, 0)
        ClassVector(1, 0)
        ClassVector(2, 0)
