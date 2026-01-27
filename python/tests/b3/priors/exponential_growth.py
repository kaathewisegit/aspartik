import pandas as pd
import pytest

import re

from aspartik.b3.parameters import Tree
from aspartik.b3.priors import ExponentialGrowth
from aspartik.data.newick import Tree as NewickTree


def test_matches_beast():
    log_path = "data/runs/respiratory/log"
    logs = pd.read_csv(log_path, sep="\t")

    trees_path = "data/runs/respiratory/trees"
    trees = open(trees_path, "r")

    for (i, row), tree in zip(logs.iterrows(), trees):
        population_size = row["population_size"]
        growth_rate = row["growth_rate"]

        tree = NewickTree(tree)
        tree = Tree.from_newick(tree)

        coalescent = ExponentialGrowth(tree, population_size, growth_rate)
        diff = abs(coalescent.probability() - row["prior:coalescent"])

        assert diff < 1e-10, f"Tree {i} has a diverging coalescent with diff {diff}"
