# ruff: noqa: E712
import math

from aspartik.math.float import sign, exponent_bits, mantissa_bits


MAX = 1.7976931348623157e308


def test_sign():
    assert sign(1.0) == False
    assert sign(-1.0) == True
    assert sign(1e10) == False
    assert sign(-1e10) == True
    assert sign(0.0) == False
    assert sign(-0.0) == True


def test_exponent_bits():
    cases = [
        (0.0, 0),
        (1.0, 1023 + 0),
        (2.0, 1023 + 1),
        (0.5, 1023 - 1),
        (10.0, 1023 + 3),
        (2**100, 1023 + 100),
        (2**100 + 77, 1023 + 100),  # non-zero mantissa
        (2**-1000, 1023 - 1000),
        (MAX, 1023 * 2),
        (math.inf, 0x7FF),
        (math.nan, 0x7FF),
        (2.0**-1022, 1),  # smallest normal
        # denormals
        (math.nextafter(0.0, 1.0), 0),
        (2**-1074, 0),
        (1e-320, 0),
    ]

    for x, expected in cases:
        assert exponent_bits(x) == expected
        # sign shouldn't matter, so we check for negated xs too
        assert exponent_bits(-x) == expected


# TODO: property testing for exponent


def test_mantissa_bits():
    cases = [
        # powers of 2
        *[(2.0**i, 0) for i in range(1000)],
        # one has already been included as 2^0
        (0.0, 0),
        (1.5, 1 << 51),
        (0.75, 1 << 51),
        (0.1, 0x999999999999A),
        # CPython implementation detail, empty quiet NaN.  If this case starts
        # failing, it should be removed
        (math.nan, 0x8000000000000),
    ]

    for x, expected in cases:
        assert mantissa_bits(x) == expected
        assert mantissa_bits(-x) == expected


# TODO: property testing for mantissa
