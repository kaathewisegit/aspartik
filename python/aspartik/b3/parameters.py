from typing import Protocol, runtime_checkable

from .._aspartik_rust_impl._b3_rust_impl import (
    Boolean as Boolean,
    Integer as Integer,
    Real as Real,
)

Parameter = Real | Integer | Boolean


@runtime_checkable
class Scalable(Protocol):
    def scale(factor: float) -> int: ...
