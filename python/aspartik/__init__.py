"""Computational biology toolkit for Python powered by Rust.

- `b3`: Bayesian phylogenetic analysis engine, analogous to BEAST2.
- `data`: biological data classes, currently only include DNA.
- `distributions`: reactive probability distributions.
- `io`: bioinformatics file formats parsers.
- `rng`: random number generator used by `b3` and `distributions`.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
