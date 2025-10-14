from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .._aspartik_rust_impl._b3_rust_impl import (
    Boolean as Boolean,
    Integer as Integer,
    Real as Real,
)
from . import Tree

Parameter = Real | Integer | Boolean


@runtime_checkable
class Scalable(Protocol):
    def scale(self, factor: float) -> int: ...


@dataclass
class Internals(Scalable):
    tree: Tree

    def scale(self, factor: float) -> int:
        self.tree.scale(factor)

        return self.tree.num_internals
