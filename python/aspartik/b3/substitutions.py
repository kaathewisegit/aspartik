from dataclasses import dataclass, field
from math import prod
from typing import ClassVar, List, SupportsFloat, Tuple

from aspartik.math import is_close


def normalize(matrix: List[List[float]], coef: float) -> List[List[float]]:
    return [[element / coef for element in row] for row in matrix]


@dataclass(slots=True)
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


@dataclass(slots=True)
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


@dataclass(slots=True)
class F81:
    dimensions: ClassVar[int] = 4
    frequencies: Tuple[float, float, float, float]
    # TODO: either a `linalg` type or something else static
    _matrix: List[List[float]]

    def __post_init__(self):
        # XXX: perhaps the frequencies should be made dynamic
        a, c, g, t = self.frequencies
        s = [
            [a - 1, c, g, t],
            [a, c - 1, g, t],
            [a, c, g - 1, t],
            [a, c, g, t - 1],
        ]
        self._matrix = normalize(s, 1 - a**2 - c**2 - g**2 - t**2)

    def get_matrix(self):
        return self._matrix


@dataclass(slots=True)
class HKY:
    dimensions: ClassVar[int] = 4
    frequencies: Tuple[float, float, float, float]
    kappa: SupportsFloat
    _cached_matrix: List[List[float]] = field(default_factory=list, init=False)
    _cached_kappa: float = field(default=0.0, init=False)

    def __post_init__(self):
        if not is_close(sum(self.frequencies), 1.0):
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


type Float6 = tuple[float, float, float, float, float, float]


@dataclass(slots=True)
class GTR:
    dimensions: ClassVar[int] = 4
    frequencies: Tuple[float, float, float, float]

    # Same as BEAST2.  Packing the fields into a vector would complicate
    # manipulating each rate individually.
    rate_ac: float = field(default=1)
    rate_ag: float = field(default=1)
    rate_at: float = field(default=1)
    rate_cg: float = field(default=1)
    rate_ct: float = field(default=1)
    rate_gt: float = field(default=1)

    _cached_matrix: List[List[float]] = field(default_factory=list, init=False)
    _cached_rates: Float6 = field(default_factory=tuple, init=False)

    def __post_init__(self):
        # TODO: deduplicate
        if not is_close(sum(self.frequencies), 1.0):
            s = sum(self.frequencies)
            raise ValueError(f"Sum of frequencies must be 1, got {s}")

        self._update_matrix()

    def _get_rates(self) -> Float6:
        return (
            float(self.rate_ac),
            float(self.rate_ag),
            float(self.rate_at),
            float(self.rate_cg),
            float(self.rate_ct),
            float(self.rate_gt),
        )

    def _update_matrix(self):
        a, c, g, t = self.frequencies
        rate_ac, rate_ag, rate_at, rate_cg, rate_ct, rate_gt = self._get_rates()

        s = [
            [0, rate_ac * c, rate_ag * g, rate_at * t],
            [rate_ac * a, 0, rate_cg * g, rate_ct * t],
            [rate_ag * a, rate_cg * c, 0, rate_ct * t],
            [rate_at * a, rate_ct * c, rate_gt * g, 0],
        ]

        for i in range(4):
            s[i][i] = -sum(s[i])

        mult = (
            rate_ac * a * c
            + rate_ag * a * g
            + rate_at * a * t
            + rate_cg * c * g
            + rate_ct * c * t
            + rate_gt * g * t
        )

        scale = 1.0 / (2.0 * mult)
        s = normalize(s, scale)

        self._cached_matrix = s
        self._cached_rates = self._get_rates()

    def get_matrix(self):
        if self._get_rates() != self._cached_rates:
            self._update_matrix()

        return self._cached_matrix
