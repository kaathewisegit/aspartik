import argparse
from typing import Optional, get_args

from aspartik.b3.config import CalculatorKind, b3_config, beast1_config, beast1_run
from aspartik.data.msa import MSA
from aspartik.io import FastaReader


def run_mcmc(
    toolkit: str,
    fasta_path: str,
    num_sequences: Optional[int],
    kind: CalculatorKind,
    length: int,
) -> Optional[float]:
    records = list(FastaReader.from_file(fasta_path))
    if num_sequences:
        if num_sequences > len(records):
            raise Exception(
                f"Not enough sequences: the alignment only has {len(records)}"
            )
        msa = MSA.from_fasta(records[:num_sequences])
    else:
        msa = MSA.from_fasta(records)

    match toolkit:
        case "b3":
            run_b3(msa, length, kind)
        case "beast1":
            run_beast1(msa, length, kind)


def run_b3(msa: MSA, length: int, kind: CalculatorKind):
    mcmc = b3_config(
        msa,
        calculator=kind,
        tree_prior="constant",
        substitution_model="HKY",
        timer=True,
        trace_path="target/bench.b3.trace",
    )
    mcmc.run(length)


def run_beast1(msa: MSA, length: int, kind: CalculatorKind):
    config = beast1_config(
        msa,
        tree_prior="constant",
        substitution_model="HKY",
        file_log_path="target/bench.beast1.log",
        tree_log_path="target/bench.beast1.trees",
        length=length,
    )

    beast1_run(config, calculator=kind)


def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("fasta_path", type=str)

    parser.add_argument("--toolkit", choices=("b3", "beast1"), default="b3")
    parser.add_argument(
        "--kind", choices=get_args(CalculatorKind.__value__), required=True
    )
    parser.add_argument("--num_sequences", type=int)
    parser.add_argument("--length", type=int, default=1_000_000)

    return parser.parse_args()


def main():
    args = parse_cli_args()

    run_mcmc(
        args.toolkit,
        args.fasta_path,
        args.num_sequences,
        args.kind,
        args.length,
    )


if __name__ == "__main__":
    main()
