from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, runtime_checkable

from .._aspartik_rust_impl._b3_rust_impl import (
    Internal as Internal,
    Leaf as Leaf,
    Proposal as Proposal,
    Real as Real,
    Tree as Tree,
)
from . import Stateful

type Node = Leaf | Internal
"""Any node of the phylogenetic tree

Used for type hints in places where there isn't a need to distinguish between
internal and leaf nodes.
"""


@runtime_checkable
class Scalable(Protocol):
    def scale(self, factor: float) -> int:
        """
        Scales all values of a parameter and returns the number of dimensions.
        """

        ...


class Weights(Stateful, Scalable):
    __slots__ = ("_value", "_backup", "_length")
    _value: list[float]
    _backup: list[float]
    _length: int

    def __init__(self, *args: float):
        self._value = list(args)
        self._backup = deepcopy(self._value)
        self._length = len(self._value)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> float:
        return self._value[i]

    def __setitem__(self, i: int, value: float):
        self._value[i] = value

    def __iter__(self):
        return iter(self._value)

    def __repr__(self) -> str:
        return repr(self._value)

    # comparison
    # XXX: deduplicate?
    def __lt__(self, other: SupportsFloat) -> bool:
        other = float(other)
        for item in self:
            if not item < other:
                return False
        return True

    def __le__(self, other: SupportsFloat) -> bool:
        other = float(other)
        for item in self:
            if not item <= other:
                return False
        return True

    def __gt__(self, other: SupportsFloat) -> bool:
        other = float(other)
        for item in self:
            if not item > other:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Weights):
            return self._value == other._value
        else:
            return False

    def __ne__(self, other: object) -> bool:
        if isinstance(other, Weights):
            return self._value != other._value
        else:
            return False

    def __ge__(self, other: SupportsFloat) -> bool:
        other = float(other)
        for item in self:
            if not item >= other:
                return False
        return True

    def scale(self, factor: float) -> int:
        for i in range(len(self)):
            self[i] *= factor
        return len(self)

    def accept(self):
        for i in range(len(self)):
            self._backup[i] = self._value[i]

    def reject(self):
        for i in range(len(self)):
            self._value[i] = self._backup[i]


@dataclass
class Internals(Scalable):
    tree: Tree

    def scale(self, factor: float) -> int:
        self.tree.scale(factor)

        return self.tree.num_internals


@dataclass
class Root:
    tree: Tree

    def __float__(self) -> float:
        return self.tree.height_of(self.tree.root)
