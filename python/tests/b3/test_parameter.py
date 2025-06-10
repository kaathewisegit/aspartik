import pickle


from aspartik.b3 import Real, Integer, Boolean


def pickle_roundtrip(obj):
    assert pickle.loads(pickle.dumps(obj)) == obj


def test_pickle():
    # basic
    r, i, b = Real(0.5), Integer(1), Boolean(True)  # noqa: F841
    pickle_roundtrip(r)
    # TODO: fix
    # pickle_roundtrip(i)
    # pickle_roundtrip(b)

    # preserves edit state
    r[0] = 1.5
    bytes = pickle.dumps(r)
    restored_r = pickle.loads(bytes)
    assert restored_r[0] == 1.5
    restored_r.reject()
    assert restored_r[0] == 0.5
