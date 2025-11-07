from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, runtime_checkable

from . import Stateful, Tree


@runtime_checkable
class Scalable(Protocol):
    def scale(self, factor: float) -> int:
        """
        Scales all values of a parameter and returns the number of dimensions.
        """

        ...


class Real(Stateful, SupportsFloat, Scalable):
    __slots__ = ("_value", "_backup")
    _value: float
    _backup: float

    def __init__(self, value: SupportsFloat):
        self._value = float(value)
        self._backup = self._value

    # comparison
    def __lt__(self, other: SupportsFloat) -> bool:
        return self._value < float(other)

    def __le__(self, other: SupportsFloat) -> bool:
        return self._value <= float(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SupportsFloat):
            return self._value == float(other)
        else:
            return False

    def __ne__(self, other: object) -> bool:
        if isinstance(other, SupportsFloat):
            return self._value != float(other)
        else:
            return False

    def __gt__(self, other: SupportsFloat) -> bool:
        return self._value > float(other)

    def __ge__(self, other: SupportsFloat) -> bool:
        return self._value >= float(other)

    # math
    def __iadd__(self, other: SupportsFloat):
        self._value += float(other)
        return self

    def __isub__(self, other: SupportsFloat):
        self._value -= float(other)
        return self

    def __imul__(self, other: SupportsFloat):
        self._value *= float(other)
        return self

    def __idiv__(self, other: SupportsFloat):
        self._value /= float(other)
        return self

    def __repr__(self) -> str:
        return repr(self._value)

    def __float__(self) -> float:
        return self._value

    def set(self, value: SupportsFloat) -> None:
        self._value = float(value)

    def scale(self, factor: float) -> int:
        self *= factor
        return 1

    def accept(self):
        self._backup = self._value

    def reject(self):
        self._value = self._backup


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
