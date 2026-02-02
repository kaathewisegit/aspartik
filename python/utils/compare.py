import pandas as pd

from aspartik.b3 import Prior
from aspartik.b3.likelihoods import Likelihood
from aspartik.b3.parameters import Parameter, Real, RealVector, Tree
from aspartik.b3.priors import ExponentialGrowth
from aspartik.data.newick import Tree as NewickTree
from aspartik.rng import RNG


def compare(
    logs_path: str,
    trees_path: str,
    *,
    parameters: dict[str, Parameter],
    priors: dict[str, Prior] = {},
    likelihoods: list[Likelihood] = [],
    beast: bool = False,
):
    if beast:
        logs = pd.read_csv(logs_path, sep="\t")
    else:
        logs = pd.read_json(logs_path, lines=True)
    newick_trees = open(trees_path, "r")

    for (i, row), newick_tree in zip(logs.iterrows(), newick_trees):
        for name, param in parameters.items():
            match param:
                case Tree():
                    param.set(Tree.from_newick(NewickTree(newick_tree)))
                case Real():
                    param.set(row[name])
                case RealVector():
                    for i, value in enumerate(row[name]):
                        param[i] = value

        for name, prior in priors.items():
            diff = abs(prior.probability() - row[name])
            assert diff < 1e-10

        for likelihood in likelihoods:
            likelihood.propose()
            diff = likelihood.likelihood() - row["likelihood"]
            assert diff < row["likelihood"] * 0.01
