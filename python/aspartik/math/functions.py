from .._aspartik_rust_impl import _math_rust_impl


# ruff: noqa: F822
__all__ = [
    # erf
    "erf",
    "erfc",
    "erf_inv",
    "erfc_inv",
    # exponential
    "ei",
    # factorial
    "factorial",
    "ln_factorial",
    "binomial",
    "ln_binomial",
]

for item in __all__:
    locals()[item] = getattr(_math_rust_impl, item)


def __dir__():
    return __all__
