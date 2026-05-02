import argparse

from .. import MCMC, TunableOperator


def print_operator_stats(mcmc: MCMC) -> None:
    print(
        f"{'Operator': <20}{'accepts%': >15}{'aborts%': >15}{'prior r.%': >15}{'tuning p.': >15}"
    )
    print("-" * 80)

    for operator, results, _, _ in mcmc.operator_statistics:
        (
            aborts,
            prior_rejects,
            accepts,
            _rejects,
        ) = results
        total = sum(results)
        accepts_share = accepts / total
        aborts_share = aborts / total
        prior_share = prior_rejects / total

        if isinstance(operator, TunableOperator):
            tuning = f"{operator.get_tuning():.2f}"
        else:
            tuning = "-"

        print(
            f"{type(operator).__name__: <20}{accepts_share: >15.0%}{aborts_share: >15.0%}{prior_share: >15.0%}{tuning: >15}"
        )


def print_operator_timings(mcmc: MCMC) -> None:
    print(
        f"{'Operator': <20}{'propose avg μs': >20}{'likelihood avg μs': >20}{'total μs': >20}"
    )
    print("-" * 80)

    for operator, results, propose, likelihood in mcmc.operator_statistics:
        (
            _aborts,
            prior_rejects,
            accepts,
            rejects,
        ) = results
        name = type(operator).__name__
        propose = (propose / (prior_rejects + accepts + rejects)).microseconds
        likelihood = (likelihood / (accepts + rejects)).microseconds
        total = propose + likelihood

        print(f"{name: <20}{propose: >20.0f}{likelihood: >20.0f}{total: >20.0f}")


def run_from_cmdline(mcmc: MCMC, default_length: int = 100_000):
    parser = argparse.ArgumentParser()

    parser.add_argument("--stats", action="store_true", help="Operator accepts/rejects")
    parser.add_argument("--timings", action="store_true", help="Operator timings")

    parser.add_argument("length", type=int, nargs="?", default=default_length)

    args = parser.parse_args()

    mcmc.run(args.length)

    if args.stats or args.timings:
        print()
    if args.stats:
        print_operator_stats(mcmc)
    if args.stats and args.timings:
        print()
    if args.timings:
        print_operator_timings(mcmc)
