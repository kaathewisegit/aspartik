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
    def scale(self, factor: float) -> int:
        """
        Scales all values of a parameter and returns the number of dimensions.
        """

        ...


@dataclass
class Internals(Scalable):
    tree: Tree

    def scale(self, factor: float) -> int:
        self.tree.scale(factor)

        return self.tree.num_internals


@dataclass
class Root(Scalable):
    tree: Tree

    def scale(self, factor: float) -> int:
        tree = self.tree
        root = tree.root

        old_height = tree.height_of(root)
        new_height = old_height * factor
        tree.set_height(root, new_height)

        if factor < 1:
            left, right = tree.children_of(root)
            left_height, right_height = tree.height_of(left), tree.height_of(right)
            if left_height > new_height or right_height > new_height:
                raise ValueError(
                    f"Scaling the root by {factor} will put it under one of its children"
                )

        return 1

    def __float__(self) -> float:
        return self.tree.height_of(self.tree.root)
