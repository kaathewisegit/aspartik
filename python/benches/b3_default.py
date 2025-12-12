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
    msa: MSA,
    duration: float,  # in seconds
    kind: Literal["cpu", "thread", "cuda"],
) -> float:
    print(f"\n# {msa.num_sequences}")

    (receiver, sender) = multiprocessing.Pipe(duplex=False)

    proc = multiprocessing.Process(target=worker, args=(msa, kind, sender))
    start_time = time.perf_counter()
    proc.start()
    time.sleep(duration)
    end_time = time.perf_counter()
    proc.terminate()

    duration = end_time - start_time
    num_steps = receiver.recv()
    speed = duration / num_steps * 1_000_000

    return speed


def worker(msa, kind, sender):
    def termination_handler(signum, frame):
        sender.send(mcmc.current_step)
        sys.exit(1)

    signal.signal(signal.SIGTERM, termination_handler)

    mcmc = default_b3_config(msa, RNG(4), 1_000_000_000, kind)

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

    records = list(FastaReader.from_file(args.file_path))
    if 2**args.magnitude > len(records):
        raise Exception(f"Not enough sequences: the alignment only has {len(records)}")

    seqs = [2**i for i in range(3, args.magnitude + 1)]
    speeds = [
        run_mcmc(MSA.from_fasta(records[:num_sequences]), args.time, args.kind)
        for num_sequences in seqs
    ]

    print(speeds)


if __name__ == "__main__":
    main()
