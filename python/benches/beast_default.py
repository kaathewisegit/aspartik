import argparse
import multiprocessing
import subprocess
import tempfile
import time
from pathlib import Path
from string import Template
from typing import Literal, Optional, get_args

from aspartik.b3.utils.beast import beast1_config
from aspartik.data.msa import MSA
from aspartik.io import FastaReader

Kind = Literal["cpu", "cuda"]


def run_mcmc(
    msa: MSA,
    length: int,
    kind: Kind,
) -> Optional[float]:
    print(f"\n# {msa.num_sequences}")

    proc = multiprocessing.Process(target=worker, args=(msa, length, kind))
    start_time = time.perf_counter()
    proc.start()
    proc.join()
    end_time = time.perf_counter()

    duration = end_time - start_time
    speed = duration / length * 1_000_000

    return speed


def worker(msa: MSA, length: int, kind: Kind):
    template = Template(Path("data/templates/beast.template").read_text())

    config = beast1_config(
        msa,
        log_path="target/bench.beast1.log",
        tree_prior="constant",
        substitution_model="HKY",
        length=length,
    )

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w+t") as tmp:
        tmp.write(config)

        args = ["beast", "-seed", "4", "-citations_off"]
        match kind:
            case "cpu":
                args.append("-beagle_CPU")
            case "cuda":
                args.append("-beagle_cuda")
        args.append(tmp.name)

        subprocess.run(args)


def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("file_path", type=str)

    parser.add_argument("--kind", choices=get_args(Kind), required=True)
    parser.add_argument("--magnitude", type=int, default=9)
    parser.add_argument("--length", type=int, default=1_000_000)

    return parser.parse_args()


def main():
    # was changed to "spawn" by default in 3.14, breaking `run_mcmc`, as MSA
    # doesn't support pickling
    multiprocessing.set_start_method("fork")

    args = parse_cli_args()

    records = list(FastaReader.from_file(args.file_path))
    if 2**args.magnitude > len(records):
        raise Exception(f"Not enough sequences: the alignment only has {len(records)}")

    seqs = [2**i for i in range(3, args.magnitude + 1)]
    speeds = [
        run_mcmc(MSA.from_fasta(records[:num_sequences]), args.length, args.kind)
        for num_sequences in seqs
    ]

    print(speeds)


if __name__ == "__main__":
    main()
