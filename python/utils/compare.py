import pandas as pd

from aspartik.b3 import Prior
from aspartik.b3.likelihoods import Likelihood
from aspartik.b3.parameters import Parameter, Real, RealVector, Tree


def compare_verify_run(
    logs_path: str,
    *,
    parameters: dict[str, Parameter],
    priors: dict[str, Prior] = {},
    likelihoods: list[Likelihood] = [],
):
    logs = pd.read_feather(logs_path)

    for i, row in logs.iterrows():
        for name, param in parameters.items():
            match param:
                case Tree():
                    param.load(row[name])
                case Real():
                    param.set(row[name])
                case RealVector():
                    for i, value in enumerate(row[name]):
                        param[i] = value

        for name, prior in priors.items():
            diff = abs(prior.probability() - row[name])
            assert diff < 1e-10

        for likelihood in likelihoods:
            diff = abs(likelihood.likelihood() - row["likelihood"])
            likelihood.reject()
            assert diff < abs(row["likelihood"]) * 0.01, (
                f"{likelihood.__class__.__name__}: {diff}"
            )
