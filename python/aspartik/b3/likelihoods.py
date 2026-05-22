"""
Felsenstein's tree likelihood calculators.
"""

from .._aspartik_rust_impl._b3_rust_impl import (
    DNALikelihood as DNALikelihood,
    GammaLikelihood as GammaLikelihood,
)

type Likelihood = DNALikelihood | GammaLikelihood
