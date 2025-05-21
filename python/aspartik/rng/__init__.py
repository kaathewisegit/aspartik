from .._aspartik_rust_impl import _rng_rust_impl

__all__ = ["RNG"]  # noqa: F822

for item in __all__:
    locals()[item] = getattr(_rng_rust_impl, item)


def __dir__():
    return __all__
