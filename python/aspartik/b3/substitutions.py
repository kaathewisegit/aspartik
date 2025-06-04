from dataclasses import dataclass
from math import prod
from typing import List, Tuple, ClassVar, SupportsFloat


def normalize(matrix: List[List[float]], coef: float) -> List[List[float]]:
    return [[element / coef for element in row] for row in matrix]


@dataclass
class JC:
    dimensions: ClassVar[int] = 4
    matrix: ClassVar[List[List[float]]] = normalize(
        [
            [-3, 1, 1, 1],
            [1, -3, 1, 1],
            [1, 1, -3, 1],
            [1, 1, 1, -3],
        ],
        3,
    )

    def get_matrix(self):
        return self.matrix


@dataclass
class K80:
    dimensions: ClassVar[int] = 4
    kappa: SupportsFloat

    def __post_init__(self):
        # TODO: check that kappa is a single-dimensional real
        pass

    def get_matrix(self):
        k = float(self.kappa)
        s = [
            [-2 - k, 1, k, 1],
            [1, -2 - k, 1, k],
            [k, 1, -2 - k, 1],
            [1, k, 1, -2 - k],
        ]
        s = normalize(s, 2 + k)

        return s


@dataclass
class F81:
    dimensions: ClassVar[int] = 4
    frequencies: Tuple[float, float, float, float]

    def __post_init__(self):
        # XXX: perhaps the frequencies should be made dynamic
        a, c, g, t = self.frequencies
        s = [
            [a - 1, c, g, t],
            [a, c - 1, g, t],
            [a, c, g - 1, t],
            [a, c, g, t - 1],
        ]
        self.matrix = normalize(s, 1 - a**2 - c**2 - g**2 - t**2)

    def get_matrix(self):
        return self.matrix


@dataclass
class HKY:
    dimensions: ClassVar[int] = 4
    frequencies: Tuple[float, float, float, float]
    kappa: SupportsFloat

    def __post_init__(self):
        # XXX: what delta should this use?
        if abs(sum(self.frequencies)) < 0.01:
            s = sum(self.frequencies)
            raise ValueError(f"Sum of frequencies must be 1, got {s}")

        self._update_matrix()

    def _update_matrix(self):
        k = float(self.kappa)

        a, c, g, t = self.frequencies
        s = [
            [0, c, k * g, t],
            [a, 0, g, k * t],
            [k * a, c, 0, t],
            [a, k * c, g, 0],
        ]

        for i in range(4):
            s[i][i] = -sum(s[i])

        purine = a + g
        pyrimidine = c + t
        scale = 1.0 / (2.0 * (purine * pyrimidine + k * prod(self.frequencies)))
        s = normalize(s, scale)

        self._cached_matrix = s
        self._cached_kappa = k

    def get_matrix(self):
        if float(self.kappa) != self._cached_kappa:
            self._update_matrix()

        return self._cached_matrix


# TODO: GTR
