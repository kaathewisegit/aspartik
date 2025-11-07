import pytest

import pickle

from aspartik.b3.parameters import Real, Weights


def pickle_roundtrip(obj):
    assert pickle.loads(pickle.dumps(obj)) == obj


# pyright: reportUnusedExpression=false
def test_eq():
    assert Real(0.1) == 0.1
    assert Real(0.1) != Real(0.2)


def test_comparison_value():
    assert Real(1e-40) < 1
    assert Real(0) <= 0
    assert Real(2.0) > 1.0


def test_pickle_roundtrip_real():
    r = Real(0.5)
    pickle_roundtrip(r)


def test_pickle_preserves_state():
    r = Real(0.5)
    r.set(1.5)
    bytes = pickle.dumps(r)
    restored_r = pickle.loads(bytes)
    assert restored_r == 1.5
    restored_r.reject()
    assert restored_r == 0.5


def test_weights_comparable():
    assert Weights(0.1, 0.2, 0.3) < 0.4
    assert Weights(0.1, 0.2, 0.3) <= 0.3
    assert Weights(0.1, 0.2, 0.3) > 0.0
    assert Weights(0.1, 0.2, 0.3) >= 0.1

    assert not Weights(0.1, 0.2, 0.3) < 0.2
    assert not Weights(0.1, 0.2, 0.3) <= 0.2
    assert not Weights(0.1, 0.2, 0.3) > 0.2
    assert not Weights(0.1, 0.2, 0.3) >= 0.2
