import argparse
import time
from typing import Optional, get_args

from aspartik.b3.utils.beast import beast1_config, beast1_run
from aspartik.b3.utils.config import CalculatorKind
from aspartik.data.msa import MSA
from aspartik.io import FastaReader


def run_mcmc(
    fasta_path: str,
    num_sequences: int,
    kind: CalculatorKind,
    length: int,
) -> Optional[float]:
    records = list(FastaReader.from_file(fasta_path))
    if num_sequences > len(records):
        raise Exception(f"Not enough sequences: the alignment only has {len(records)}")
    msa = MSA.from_fasta(records[:num_sequences])

    start_time = time.perf_counter()
    run_bench(msa, length, kind)
    end_time = time.perf_counter()

    duration = end_time - start_time
    speed = duration / length * 1_000_000

    return speed


def run_bench(msa: MSA, length: int, kind: CalculatorKind):
    config = beast1_config(
        msa,
        log_path="target/bench.beast1.log",
        tree_prior="constant",
        substitution_model="HKY",
        length=length,
    )

    beast1_run(config, calculator=kind)


def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("fasta_path", type=str)

    parser.add_argument(
        "--kind", choices=get_args(CalculatorKind.__value__), required=True
    )
    parser.add_argument("--num_sequences", type=int)
    parser.add_argument("--length", type=int, default=1_000_000)

    return parser.parse_args()


def main():
    args = parse_cli_args()

    print(run_mcmc(args.fasta_path, args.num_sequences, args.kind, args.length))


if __name__ == "__main__":
    main()
