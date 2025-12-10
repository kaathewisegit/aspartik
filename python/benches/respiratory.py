from utils.b3_config import default as default_b3_config

import time
from typing import Literal, Optional

from aspartik.data.msa import MSA
from aspartik.io import FastaReader
from aspartik.rng import RNG


def run_mcmc(
    data_path: str,
    length: int,
    num_sequences: Optional[int] = None,
    kind: Literal["thread", "cuda"] = "thread",
) -> float:
    records = list(FastaReader.from_file(data_path))
    if num_sequences is None:
        num_sequences = len(records)
    msa = MSA.from_fasta(records[:num_sequences])
    mcmc = default_b3_config(msa, RNG(4), length, kind)

    start_time = time.perf_counter()
    mcmc.run()
    end_time = time.perf_counter()

    duration = end_time - start_time

    speed = duration / length * 1_000_000

    return speed


if __name__ == "__main__":
    seqs = [8, 16, 32, 64, 128, 256, 512]
    speeds = [
        run_mcmc("data/alignments/respiratory.fasta", 1_000_000, num_sequences, "cuda")
        for num_sequences in seqs
    ]

    print(speeds)
