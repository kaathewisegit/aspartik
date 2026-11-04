import polars as pl

from aspartik.b3 import Prior
from aspartik.b3.likelihoods import Likelihood
from aspartik.b3.parameters import DatedTree, IntVector, Parameter, Real, RealVector
from aspartik.data.tree import TreeBuilder as NewickTree


def replicate_b3(
    trace_path: str,
    parameters: dict[str, Parameter],
    priors: dict[str, Prior] = {},
    likelihoods: list[Likelihood] = [],
):
    trace = pl.read_ipc(trace_path)

    for batch in trace.iter_slices(n_rows=100):
        for i in range(len(batch)):
            for name, param in parameters.items():
                match param:
                    case DatedTree():
                        parents = batch[f"{name}:parents"][i].to_numpy()
                        heights = batch[f"{name}:heights"][i].to_numpy()
                        param.load_parts(parents, heights)
                    case Real():
                        param.set(batch[name][i])
                    case RealVector():
                        for j, value in enumerate(batch[name][i]):
                            param[j] = value

            expected = {prior: batch[prior][i] for prior in priors}
            expected["likelihood"] = batch["likelihood"][i]
            _check(expected, priors, likelihoods)


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
                case DatedTree():
                    param.load_newick(NewickTree.from_newick(tree))
                case Real():
                    param.set(float(row[name]))
                case RealVector():
                    for i in range(len(param)):
                        param[i] = float(row[f"{name}{i}"])
                case IntVector():
                    for i in range(len(param)):
                        param[i] = int(row[f"{name}{i}"])
                case _:
                    raise TypeError(
                        f"Parameter {param.__class__.__name__} isn't supported"
                    )

        _check(row, priors, likelihoods)


def replicate_beast2(
    log_path: str,
    parameters: dict[str, Parameter],
    trees_path: str | None = None,
    priors: dict[str, Prior] = {},
    likelihoods: list[Likelihood] = [],
):
    log = pl.read_csv(log_path, separator="\t", comment_prefix="#")
    trees = open(trees_path, "r") if trees_path is not None else None

    try:
        for row in log.iter_rows(named=True):
            for name, param in parameters.items():
                match param:
                    case DatedTree():
                        tree = next(trees) if trees is not None else row[name]
                        param.load_newick(NewickTree.from_newick(tree))
                    case Real():
                        param.set(float(row[name]))
                    case RealVector():
                        for i in range(len(param)):
                            param[i] = float(row[f"{name}{i + 1}"])
                    case IntVector():
                        for i in range(len(param)):
                            param[i] = int(row[f"{name}{i + 1}"])
                    case _:
                        raise TypeError(
                            f"Parameter {param.__class__.__name__} isn't supported"
                        )

            _check(row, priors, likelihoods)
    finally:
        if trees is not None:
            trees.close()


def _check(row, priors: dict[str, Prior] = {}, likelihoods: list[Likelihood] = []):
    for name, prior in priors.items():
        expected = float(row[name])
        got = prior.probability()
        diff = abs(expected - got)
        rel = diff / abs(expected)
        assert rel < 1e-12, f"got {got}, expected {expected}, prior: {prior}"

    for likelihood in likelihoods:
        expected = float(row["likelihood"])
        got = likelihood.likelihood()
        diff = abs(got - expected)
        assert diff < 1e-9 * abs(expected), (
            f"{likelihood.__class__.__name__}: got {got}, expected {expected} ({diff / abs(expected)})"
        )
        likelihood.reject()
