import pickle
import pytest


from aspartik.b3 import Real, Integer, Boolean


def pickle_roundtrip(obj):
    assert pickle.loads(pickle.dumps(obj)) == obj


# pyright: reportUnusedExpression=false
def test_eq():
    assert Real(0.5, 0.1) == Real(0.5, 0.1)
    assert Real(0.1) != Real(0.2)
    assert Integer(1) == Integer(1)
    assert Boolean(True) != Boolean(False)

    with pytest.raises(ValueError) as error:
        Integer(1) == Integer(1, 2)
    assert "Can't compare parameters of different lengths: 1 and 2" in str(error.value)

    with pytest.raises(ValueError):
        Boolean(True) == Boolean(True, True)

    with pytest.raises(ValueError):
        Real(0.5, 0.5, 0.5) == Real(0.5, 0.5, 0.5, 0.5)

    with pytest.raises(TypeError):
        Boolean(True) == Real(0.1)

    with pytest.raises(TypeError):
        Real(0.1) == 0.1

    with pytest.raises(TypeError):
        Integer(1, 1) == 1


# XXX: do these semantics make sense?
def test_comparison_elements():
    assert Real(100.5, 99.5) > Real(12.5, 99)
    assert Integer(1, 2) > Integer(-100, -1000)

    assert Real(0.3) >= Real(0.3)
    assert Integer(1, 2, 3) >= Integer(1, 2, 3)

    assert Real(0) < Real(1e40)
    assert Integer(-1000, 0) < Integer(-999, 1)
    assert not Integer(-1000, 0) < Integer(-999, 0)

    assert Real(1e-40) <= Real(1e-40)
    assert Integer(0, 1) <= Integer(0, 2)


def test_comparison_value():
    assert Real(1e-40) < 1
    assert Real(1, 2, 3, 0.5) < 3.5
    assert Integer(100) < 101
    assert Integer(1, 2, 3) < 4

    assert Real(0) <= 0
    assert Real(0.25, 0.25, 0.25, 0.25) <= 0.25
    assert Integer(1) <= 1
    assert Integer(0, 1, 0, 1) <= 1

    assert Real(2.0) > 1.0
    assert Real(0.8, 0.1, 0.05, 0.05) > 0
    assert Integer(1) > 0
    assert Integer(1, 2, 3) > 0

    assert Real(2.0, 1.0) >= 1.0
    assert Real(0.25, 0.25, 0.25, 0.25) >= 0.25
    assert Integer(1) >= 1
    assert Integer(1, 2, 3) >= 1

    with pytest.raises(TypeError) as error:
        # the type error is deliberate
        Integer(1, 2, 3) > 0.5  # type: ignore
    assert "Integer can only be compared to other instances or to int" in str(
        error.value
    )


def test_pickle_roundtrip_basic():
    r, i, b = Real(0.5), Integer(1), Boolean(True)
    pickle_roundtrip(r)
    pickle_roundtrip(i)
    pickle_roundtrip(b)


def test_pickle_roundtrip_multidimensional():
    r, i, b = Real(-0.5, 0.0, 0.5), Integer(1, 2, 3), Boolean(True, True, False, True)
    pickle_roundtrip(r)
    pickle_roundtrip(i)
    pickle_roundtrip(b)


def test_pickle_preserves_state():
    r = Real(0.5)
    r[0] = 1.5
    bytes = pickle.dumps(r)
    restored_r = pickle.loads(bytes)
    assert restored_r[0] == 1.5
    restored_r.reject()
    assert restored_r[0] == 0.5
