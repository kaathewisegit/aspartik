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


# TODO: other CompareOp methods


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
