from .._aspartik_rust_impl import _data_rust_impl

__all__ = ["MSA"]  # noqa: F822

for item in __all__:
    locals()[item] = getattr(_data_rust_impl, item)


def __dir__():
    return __all__
