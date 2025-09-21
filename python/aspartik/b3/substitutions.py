from dataclasses import dataclass, field
from math import prod
from typing import ClassVar, Protocol, SupportsFloat

from aspartik.math import is_close

type Float4 = tuple[float, float, float, float]
type Float6 = tuple[float, float, float, float, float, float]
# TODO: either a `linalg` type or something else static
type Matrix = list[list[float]]


def normalize(matrix: Matrix, coef: float) -> Matrix:
    return [[element / coef for element in row] for row in matrix]


def check_frequencies(f: Float4) -> None:
    if not is_close(sum(f), 1.0):
        raise ValueError(f"The sum of frequencies must be 1, got {sum(f)}")


class Substitution(Protocol):
    """
    A substitution model

    Substitution models describe the chance of site state transitions.  They
    are used by likelihood calculators.
    """

    dimensions: ClassVar[int]
    """
    The number of states the substitution model has.

    Currently only DNA models with 4 dimensions are supported.
    """

    def get_matrix(self) -> Matrix:
        """
        Returns the current substitution matrix

        "Current" here means that it's the matrix corresponding to the values
        the substitution's arguments have at the time of the `get_matrix` call.

        MCMC calls `get_matrix` via likelihood calculations after the operator
        proposal.  So, a substitution should calculate the matrix reactively
        based on the updated argument values.

        **NOTE**: currently, all matrices retuned by this method must be
        symmetrical.
        """
        ...


@dataclass(slots=True)
class JC(Substitution):
    """
    Jukes-Cantor

    A simple model with equal state transition rates.

    Jukes and Cantor 1969, Evolution of Protein Molecules,
    <https://doi.org/10.1016/b978-1-4832-3211-9.50009-7>.
    """

    dimensions: ClassVar[int] = 4
    matrix: ClassVar[Matrix] = normalize(
        [
            [-3, 1, 1, 1],
            [1, -3, 1, 1],
            [1, 1, -3, 1],
            [1, 1, 1, -3],
        ],
        3,
    )

    def get_matrix(self) -> Matrix:
        return self.matrix


@dataclass(slots=True)
class K80(Substitution):
    """Kimura 80

    Equal base frequencies (A/C/G/T) with different transition (keeps
    purines/pyrimidines) and transversion (purine to pyrimidine and visa
    versa).

    Kimura 1980, A simple method for estimating evolutionary rates of base
    substitutions through comparative studies of nucleotide sequences,
    <https://doi.org/10.1007/BF01731581>.
    """

    dimensions: ClassVar[int] = 4
    kappa: SupportsFloat
    """
    A transition is taken to be kappa times more likely than a transversion.
    """

    def __post_init__(self):
        # TODO: check that kappa is a single-dimensional real
        pass

    def get_matrix(self) -> Matrix:
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
class F81(Substitution):
    """
    Felsenstein 1981

    A model which depends on base frequencies, but doesn't distinguish between
    transitions and transversions.

    Felsenstein 1981, Evolutionary Trees from DNA Sequences: A Maximum
    Likelihood Approach, <https://doi.org/10.1007/BF01734359>.
    """

    dimensions: ClassVar[int] = 4
    frequencies: Float4
    """A tuple of frequencies in order `(A, C, G, T)`"""
    _matrix: Matrix

    def __post_init__(self):
        check_frequencies(self.frequencies)

        # XXX: are dynamic frequencies used in real-world analysis?
        a, c, g, t = self.frequencies
        s = [
            [a - 1, c, g, t],
            [a, c - 1, g, t],
            [a, c, g - 1, t],
            [a, c, g, t - 1],
        ]
        self._matrix = normalize(s, 1 - a**2 - c**2 - g**2 - t**2)

    def get_matrix(self) -> Matrix:
        return self._matrix


@dataclass(slots=True)
class HKY(Substitution):
    """
    Hasegawa et al. 1985

    A model which can be thought of as a combination of K80 and F81: both base
    rates and transition/transversion ratio are configurable.

    Hasegawa et al. 1985, Dating of the human-ape splitting by a molecular
    clock of mitochondrial DNA, <https://doi.org/10.1007/BF02101694>.
    """

    dimensions: ClassVar[int] = 4
    frequencies: Float4
    """A tuple of frequencies in order `(A, C, G, T)`"""
    kappa: SupportsFloat
    """Transition rate divided transversion rate"""
    _cached_matrix: Matrix = field(default_factory=list, init=False)
    _cached_kappa: float = field(default=0.0, init=False)

    def __post_init__(self):
        check_frequencies(self.frequencies)

        self._update_matrix()

    def _update_matrix(self) -> None:
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

    def get_matrix(self) -> Matrix:
        if float(self.kappa) != self._cached_kappa:
            self._update_matrix()

        return self._cached_matrix


@dataclass(slots=True)
class GTR(Substitution):
    """
    General time reversible

    A general version of a 4-dimensional DNA substitution model, which allows
    creating any symmetrical substitution matrix.

    Lanave et al. 1984, A new method for calculating evolutionary substitution
    rates, <https://doi.org/10.1007/BF02101990>.
    """

    dimensions: ClassVar[int] = 4
    frequencies: Float4
    """A tuple of frequencies in order `(A, C, G, T)`"""

    # Same as BEAST2.  Packing the fields into a vector would complicate
    # manipulating each rate individually.
    rate_ac: float = field(default=1)
    """
    Transition rate between adenine and cytosine

    It's symmetrical (A -> C, C -> A), same as all of the following rate
    arguments.
    """
    rate_ag: float = field(default=1)
    rate_at: float = field(default=1)
    rate_cg: float = field(default=1)
    rate_ct: float = field(default=1)
    rate_gt: float = field(default=1)

    _cached_matrix: Matrix = field(default_factory=list, init=False)
    _cached_rates: Float6 = field(default_factory=tuple, init=False)

    def __post_init__(self):
        check_frequencies(self.frequencies)

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

    def _update_matrix(self) -> None:
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

    def get_matrix(self) -> Matrix:
        if self._get_rates() != self._cached_rates:
            self._update_matrix()

        return self._cached_matrix
