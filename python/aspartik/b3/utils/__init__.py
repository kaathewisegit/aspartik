from .. import MCMC


def print_operator_stats(mcmc: MCMC) -> None:
    print(f"{'Operator': <20}{'#accepts': >20}{'#rejects': >20}{'%accepts': >15}")
    print("-" * 75)

    for operator, accepts, rejects in mcmc.operator_statistics:
        share = accepts / (accepts + rejects)
        print(
            f"{type(operator).__name__: <20}{accepts: >20}{rejects: >20}{share: >15.0%}"
        )
