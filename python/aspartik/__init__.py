"""Computational biology toolkit for Python powered by Rust.

- `b3`: Bayesian phylogenetic analysis engine, analogous to BEAST2.
- `data`: biological data classes, currently only include DNA.
- `io`: bioinformatics file formats parsers.
- `rng`: random number generator used by `b3` and `stats`.
- `stats`: statistical functions.
"""

from . import b3 as b3
from . import data as data
from . import io as io
from . import rng as rng
from . import stats as stats
