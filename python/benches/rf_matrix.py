import argparse
import statistics
from time import perf_counter

from aspartik.data.tree import BinaryTree, robinson_foulds_matrix
from aspartik.rng import RNG


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def num_leaves(value):
    value = int(value)
    if value < 2:
        raise argparse.ArgumentTypeError("expected at least two leaves")
    return value


def validate_matrix(matrix, tree_count, leaf_count):
    if len(matrix) != tree_count:
        raise RuntimeError("RF matrix has the wrong number of rows")
    maximum = 2 * (leaf_count - 2)
    checksum = 0
    observed_maximum = 0
    for first, row in enumerate(matrix):
        if len(row) != tree_count:
            raise RuntimeError("RF matrix has the wrong number of columns")
        if row[first] != 0:
            raise RuntimeError("RF matrix diagonal is not zero")
        for second, distance in enumerate(row):
            if distance != matrix[second][first]:
                raise RuntimeError("RF matrix is not symmetric")
            if distance > maximum or distance % 2 != 0:
                raise RuntimeError("RF matrix contains an invalid distance")
            checksum += distance
            observed_maximum = max(observed_maximum, distance)
    return observed_maximum, checksum


def benchmark(tree_count, leaf_count, seed, repeats):
    timings = []
    print(f"trees={tree_count} leaves={leaf_count} seed={seed} repeats={repeats}")
    print("repeat\tgeneration_s\trf_matrix_s\tmax_rf\tchecksum")
    for repeat in range(1, repeats + 1):
        rng = RNG(seed)
        started = perf_counter()
        trees = [BinaryTree.random(leaf_count, rng) for _ in range(tree_count)]
        generated = perf_counter()
        matrix = robinson_foulds_matrix(trees)
        finished = perf_counter()
        maximum, checksum = validate_matrix(matrix, tree_count, leaf_count)
        generation_seconds = generated - started
        matrix_seconds = finished - generated
        timings.append((generation_seconds, matrix_seconds))
        print(
            f"{repeat}\t{generation_seconds:.6f}\t{matrix_seconds:.6f}"
            f"\t{maximum}\t{checksum}"
        )

    generation_median = statistics.median(value[0] for value in timings)
    matrix_median = statistics.median(value[1] for value in timings)
    print(f"median\t{generation_median:.6f}\t{matrix_median:.6f}\t-\t-")


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=positive_int)
    parser.add_argument("--num-leaves", type=num_leaves, default=100)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--repeats", type=positive_int, default=3)
    return parser.parse_args()


def main():
    args = parse_cli_args()
    benchmark(args.n, args.num_leaves, args.seed, args.repeats)


if __name__ == "__main__":
    main()
