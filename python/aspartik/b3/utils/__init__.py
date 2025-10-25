from .. import MCMC


def print_operator_stats(mcmc: MCMC) -> None:
    print(f"{'Operator': <20}{'%accepts': >20}{'%aborts': >20}{'%prior rejects': >20}")
    print("-" * 80)

    for operator, results, _, _ in mcmc.operator_statistics:
        (
            unconditional_accepts,
            unconditional_rejects,
            prior_rejects,
            accepts,
            rejects,
        ) = results
        total = sum(results)
        accepts_share = accepts / total
        aborts_share = unconditional_rejects / total
        prior_share = prior_rejects / total
        print(
            f"{type(operator).__name__: <20}{accepts_share: >20.0%}{aborts_share: >20.0%}{prior_share: >20.0%}"
        )


def print_operator_timings(mcmc: MCMC) -> None:
    print(
        f"{'Operator': <20}{'%Aborts': >20}{'propose avg μs': >20}{'likelihood avg μs': >20}{'total μs': >20}"
    )
    print("-" * 80)

    for operator, results, propose, likelihood in mcmc.operator_statistics:
        (
            unconditional_accepts,
            unconditional_rejects,
            prior_rejects,
            accepts,
            rejects,
        ) = results
        name = type(operator).__name__
        propose = (propose / (prior_rejects + accepts + rejects)).microseconds
        likelihood = (likelihood / (accepts + rejects)).microseconds
        total = propose + likelihood

        print(f"{name: <20}{propose: >20.0f}{likelihood: >20.0f}{total: >20.0f}")
