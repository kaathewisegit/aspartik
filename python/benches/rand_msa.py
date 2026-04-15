import argparse
from typing import Literal, Optional

from aspartik.b3.config import MCMCConfig
from aspartik.data.msa import MSA
from aspartik.rng import RNG


def run_mcmc(
    toolkit: str,
    num_sequences: int,
    num_sites: int,
    kind: Literal["cpu", "cuda"],
    length: int,
    seed: int = 4,
) -> Optional[float]:
    rng = RNG(seed)
    msa = MSA.random(
        num_sequences, num_sites, [str(i) for i in range(num_sequences)], rng
    )

    config = MCMCConfig(
        msa,
        calculator=kind,
        tree_prior="constant",
        substitution_model="HKY",
        trace_path="target/rand_msa.trace",
        trees_path="target/rand_msa.trees",
        timer=True,
    )

    match toolkit:
        case "b3":
            config.b3_make_and_run()
        case "beast1":
            config.beast1_make_and_run()


def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--toolkit", choices=("b3", "beast1"), default="b3")
    parser.add_argument("--kind", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--num-sites", type=int, required=True)
    parser.add_argument("--length", type=int, default=1_000_000)

    return parser.parse_args()


def main():
    args = parse_cli_args()

    run_mcmc(
        args.toolkit,
        args.num_sequences,
        args.num_sites,
        args.kind,
        args.length,
    )


if __name__ == "__main__":
    main()
