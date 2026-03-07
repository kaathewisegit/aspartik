"""
A benchmark for estimating non-calculator elements of the analysis
"""

import argparse
import time

from aspartik.b3.config import b3_config
from aspartik.b3.utils import print_operator_timings
from aspartik.data import DNASeq
from aspartik.data.msa import MSA


def run_mcmc(num_leaves: int, length: int):
    msa = MSA([str(i) for i in range(num_leaves)], [DNASeq("A")] * num_leaves)
    mcmc = b3_config(msa, tree_prior="constant", substitution_model="HKY")

    start = time.perf_counter()
    mcmc.run(length)
    end = time.perf_counter()
    print_operator_timings(mcmc)

    return (end - start) / (length / 1_000_000)


def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_leaves", type=int, default=2**15)
    parser.add_argument("--length", type=int, default=10_000)

    return parser.parse_args()


def main():
    args = parse_cli_args()

    speed = run_mcmc(args.num_leaves, args.length)

    print(speed)


if __name__ == "__main__":
    main()
