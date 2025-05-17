"""Computational biology toolkit for Python powered by Rust.

Aspartik contains a suit of programs developed for the sake of the `b3`
Bayesian phylogenetic inference engine.  Most of them are currently very raw
and only expose the functionality required by `b3`, but I hope that as
development continues they'll become broadly useful.

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
