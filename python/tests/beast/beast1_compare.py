"""
<https://beast.community/first_tutorial>
"""

import pytest

from aspartik.b3.utils.compare import compare_beast1


@pytest.mark.manual
def test_yule_hky_small():
    compare_beast1(
        "data/alignments/apes.fasta",
        2_000_000,
        model="HKY",
        tree_prior="yule",
    )


@pytest.mark.manual
def test_constant_hky_small():
    compare_beast1(
        "data/alignments/apes.fasta",
        2_000_000,
        model="HKY",
        tree_prior="constant",
    )
