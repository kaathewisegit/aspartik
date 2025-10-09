from .. import MCMC


def print_operator_stats(mcmc: MCMC) -> None:
    print(f"{'Operator': <20}{'#accepts': >20}{'#rejects': >20}{'%accepts': >20}")
    print("-" * 80)

    for operator, accepts, rejects, _, _ in mcmc.operator_statistics:
        share = accepts / (accepts + rejects)
        print(
            f"{type(operator).__name__: <20}{accepts: >20}{rejects: >20}{share: >20.0%}"
        )


def print_operator_timings(mcmc: MCMC) -> None:
    print(
        f"{'Operator': <20}{'propose avg μs': >20}{'likelihood avg μs': >20}{'total μs': >20}"
    )
    print("-" * 80)

    for operator, accepts, rejects, propose, likelihood in mcmc.operator_statistics:
        total = accepts + rejects
        name = type(operator).__name__
        propose = (propose / total).microseconds
        likelihood = (likelihood / total).microseconds
        total = propose + likelihood

        print(f"{name: <20}{propose: >20.0f}{likelihood: >20.0f}{total: >20.0f}")
