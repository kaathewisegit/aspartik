from collections.abc import Sequence
from dataclasses import dataclass
from typing import SupportsFloat

@dataclass
class JC:
    """
    Jukes-Cantor

    A simple model with equal state transition rates.

    Jukes and Cantor 1969, Evolution of Protein Molecules,
    <https://doi.org/10.1016/b978-1-4832-3211-9.50009-7>.
    """

@dataclass
class K80:
    """Kimura 80

    Equal base frequencies (A/C/G/T) with different transition (keeps
    purines/pyrimidines) and transversion (purine to pyrimidine and visa
    versa).

    Kimura 1980, A simple method for estimating evolutionary rates of base
    substitutions through comparative studies of nucleotide sequences,
    <https://doi.org/10.1007/BF01731581>.
    """

    kappa: SupportsFloat
    """
    A transition is taken to be kappa times more likely than a transversion.
    """

@dataclass
class HKY:
    """
    Hasegawa et al. 1985

    A model which can be thought of as a combination of K80 and F81: both base
    rates and transition/transversion ratio are configurable.

    Hasegawa et al. 1985, Dating of the human-ape splitting by a molecular
    clock of mitochondrial DNA, <https://doi.org/10.1007/BF02101694>.
    """

    frequencies: Sequence[float]
    kappa: SupportsFloat
