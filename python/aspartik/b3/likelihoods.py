"""
Felsenstein's tree likelihood calculators.
"""

from .._aspartik_rust_impl._b3_rust_impl import (
    CompoundLikelihood as CompoundLikelihood,
    DNALikelihood as DNALikelihood,
    GammaLikelihood as GammaLikelihood,
    HeteroLikelihood as HeteroLikelihood,
)

type Likelihood = (
    DNALikelihood | GammaLikelihood | CompoundLikelihood | HeteroLikelihood
)
