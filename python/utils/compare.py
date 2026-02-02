import pandas as pd

from aspartik.b3 import Prior
from aspartik.b3.likelihoods import Likelihood
from aspartik.b3.parameters import Parameter, Real, RealVector, Tree
from aspartik.rng import RNG


def compare(
    logs_path: str,
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

    for i, row in logs.iterrows():
        print(row)
        print(row["kappa"])
        print(row["frequencies"])
        print(row["tree:newick"])
        print(row["tree"])
        for name, param in parameters.items():
            match param:
                case Tree():
                    param.set(Tree.from_json(row[name]))
                    print(param.newick(internal_ids=True))
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
            print(likelihood.likelihood(), row["likelihood"])
            diff = abs(likelihood.likelihood() - row["likelihood"])
            assert diff < 0.5
            # assert diff < abs(row["likelihood"]) * 0.0000000001
