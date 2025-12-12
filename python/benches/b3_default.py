from utils.b3_config import default as default_b3_config

import argparse
import multiprocessing
import signal
import sys
import time
from typing import Literal, Optional

from aspartik.data.msa import MSA
from aspartik.io import FastaReader
from aspartik.rng import RNG


def run_mcmc(
    data_path: str,
    duration: float,  # in seconds
    kind: Literal["cpu", "thread", "cuda"],
    num_sequences: int,
) -> float:
    print(f"\n# {num_sequences}")

    records = list(FastaReader.from_file(data_path))
    if num_sequences > len(records):
        raise Exception("Not enough sequences: the alignment only has {len(records)}")

    msa = MSA.from_fasta(records[:num_sequences])
    mcmc = default_b3_config(msa, RNG(4), 1_000_000_000, kind)

    (receiver, sender) = multiprocessing.Pipe(duplex=False)

    proc = multiprocessing.Process(target=worker, args=(mcmc, sender))
    start_time = time.perf_counter()
    proc.start()
    time.sleep(duration)
    end_time = time.perf_counter()
    proc.terminate()

    duration = end_time - start_time
    num_steps = receiver.recv()
    speed = duration / num_steps * 1_000_000

    return speed


def worker(mcmc, sender):
    def termination_handler(signum, frame):
        sender.send(mcmc.current_step)
        sys.exit(1)

    signal.signal(signal.SIGTERM, termination_handler)

    try:
        mcmc.run()
    except:
        # silence the exception
        pass


def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("file_path", type=str)

    parser.add_argument("--kind", choices=["cpu", "thread", "cuda"], required=True)
    parser.add_argument("--magnitude", type=int, default=9)
    parser.add_argument("--time", type=int, default=60, help="In seconds")

    return parser.parse_args()


def main():
    args = parse_cli_args()

    seqs = [2**i for i in range(3, args.magnitude + 1)]
    speeds = [
        run_mcmc(args.file_path, args.time, args.kind, num_sequences)
        for num_sequences in seqs
    ]

    print(speeds)


if __name__ == "__main__":
    main()
