from .._aspartik_rust_impl import _math_rust_impl

__all__ = [
    # Rust
    "is_close",
    # Python
    "functions",
]


for item in __all__[:1]:
    locals()[item] = getattr(_math_rust_impl, item)


def __dir__():
    return __all__
