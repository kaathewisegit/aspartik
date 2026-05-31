import polars as pl

from aspartik.b3 import Prior
from aspartik.b3.likelihoods import Likelihood
from aspartik.b3.parameters import Parameter, Real, RealVector, Tree
from aspartik.data.newick import Tree as NewickTree


def replicate_b3(
    trace_path: str,
    parameters: dict[str, Parameter],
    priors: dict[str, Prior] = {},
    likelihoods: list[Likelihood] = [],
):
    trace = pl.read_ipc(trace_path)

    for row in trace.iter_rows(named=True):
        for name, param in parameters.items():
            match param:
                case Tree():
                    param.load(row[name])
                case Real():
                    param.set(row[name])
                case RealVector():
                    print(row[name])
                    for i, value in enumerate(row[name]):
                        param[i] = value

        _check(row, priors, likelihoods)


def replicate_beast1(
    log_path: str,
    trees_path: str,
    parameters: dict[str, Parameter],
    priors: dict[str, Prior] = {},
    likelihoods: list[Likelihood] = [],
):
    log = pl.read_csv(log_path, separator="\t", skip_lines=3)
    trees = open(trees_path, "r")

    for row, tree in zip(log.iter_rows(named=True), trees):
        for name, param in parameters.items():
            match param:
                case Tree():
                    param.load_newick(NewickTree(tree))
                case Real():
                    param.set(float(row[name]))
                case RealVector():
                    for i in range(len(param)):
                        param[i] = float(row[f"{name}{i}"])

        _check(row, priors, likelihoods)


def _check(row, priors: dict[str, Prior] = {}, likelihoods: list[Likelihood] = []):
    for name, prior in priors.items():
        diff = abs(prior.probability() - float(row[name]))
        assert diff < 1e-5

    for likelihood in likelihoods:
        expected = float(row["likelihood"])
        diff = abs(likelihood.likelihood() - expected)
        assert diff < 0.001 * abs(expected), (
            f"{likelihood.__class__.__name__}: {diff} ({diff / abs(expected):.2%})"
        )
        likelihood.reject()
