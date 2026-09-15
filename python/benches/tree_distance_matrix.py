import argparse
import math
import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from aspartik.data.tree import BinaryTree, Tree, robinson_foulds_matrix
from aspartik.rng import RNG

Metric = Literal["rf", "branch-score", "triplet"]
METRICS = ("rf", "branch-score", "triplet")


@dataclass(frozen=True)
class BenchmarkRun:
    generation_seconds: float
    matrix_seconds: float
    maximum: int | float
    checksum: int | float


@dataclass(frozen=True)
class BenchmarkResult:
    metric: Metric
    tree_count: int
    leaf_count: int
    seed: int
    runs: tuple[BenchmarkRun, ...]

    @property
    def generation_median(self):
        return statistics.median(run.generation_seconds for run in self.runs)

    @property
    def matrix_median(self):
        return statistics.median(run.matrix_seconds for run in self.runs)


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


def pairwise_matrix(trees, distance):
    matrix = [[0 for _ in trees] for _ in trees]
    for first in range(len(trees)):
        for second in range(first):
            value = distance(trees[first], trees[second])
            matrix[first][second] = value
            matrix[second][first] = value
    return matrix


def distance_matrix(metric, trees):
    if metric == "rf":
        return robinson_foulds_matrix(trees)
    if metric == "branch-score":
        return pairwise_matrix(trees, BinaryTree.branch_score)
    if metric == "triplet":
        return pairwise_matrix(trees, BinaryTree.triplet_distance)
    raise ValueError(f"unknown distance metric: {metric}")


def random_tree(metric, leaf_count, rng):
    tree = BinaryTree.random(leaf_count, rng)
    if metric != "branch-score":
        return tree

    builder = Tree.from_newick(tree.to_newick())
    for node in builder.nodes():
        if node != builder.root:
            builder.set_edge_length(node, rng.random_float(0.1, 1.0))
    return builder.to_binary()


def validate_matrix(metric, matrix, tree_count, leaf_count):
    if len(matrix) != tree_count:
        raise RuntimeError("distance matrix has the wrong number of rows")
    checksum = 0
    observed_maximum = 0
    for first, row in enumerate(matrix):
        if len(row) != tree_count:
            raise RuntimeError("distance matrix has the wrong number of columns")
        if row[first] != 0:
            raise RuntimeError("distance matrix diagonal is not zero")
        for second, distance in enumerate(row):
            if distance != matrix[second][first]:
                raise RuntimeError("distance matrix is not symmetric")
            if distance < 0 or not math.isfinite(distance):
                raise RuntimeError("distance matrix contains an invalid distance")
            if metric == "rf":
                maximum = 2 * (leaf_count - 2)
                if distance > maximum or distance % 2 != 0:
                    raise RuntimeError("RF matrix contains an invalid distance")
            elif metric == "triplet" and distance > math.comb(leaf_count, 3):
                raise RuntimeError("triplet matrix contains an invalid distance")
            checksum += distance
            observed_maximum = max(observed_maximum, distance)
    return observed_maximum, checksum


def run_benchmark(
    tree_count,
    leaf_count=100,
    seed=4,
    repeats=3,
    metric: Metric = "rf",
):
    if tree_count < 1:
        raise ValueError("expected at least one tree")
    if leaf_count < 2:
        raise ValueError("expected at least two leaves")
    if repeats < 1:
        raise ValueError("expected at least one repeat")
    if metric not in METRICS:
        raise ValueError(f"unknown distance metric: {metric}")

    runs = []
    for _ in range(repeats):
        rng = RNG(seed)
        started = perf_counter()
        trees = [random_tree(metric, leaf_count, rng) for _ in range(tree_count)]
        generated = perf_counter()
        matrix = distance_matrix(metric, trees)
        finished = perf_counter()
        maximum, checksum = validate_matrix(metric, matrix, tree_count, leaf_count)
        runs.append(
            BenchmarkRun(
                generated - started,
                finished - generated,
                maximum,
                checksum,
            )
        )

    return BenchmarkResult(metric, tree_count, leaf_count, seed, tuple(runs))


def format_number(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def print_result(result):
    print(
        f"metric={result.metric} trees={result.tree_count} "
        f"leaves={result.leaf_count} seed={result.seed} "
        f"repeats={len(result.runs)}"
    )
    print("repeat\tgeneration_s\tdistance_matrix_s\tmaximum\tchecksum")
    for repeat, run in enumerate(result.runs, 1):
        print(
            f"{repeat}\t{run.generation_seconds:.6f}"
            f"\t{run.matrix_seconds:.6f}\t{format_number(run.maximum)}"
            f"\t{format_number(run.checksum)}"
        )
    print(f"median\t{result.generation_median:.6f}\t{result.matrix_median:.6f}\t-\t-")


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=positive_int)
    parser.add_argument("--metric", choices=METRICS, default="rf")
    parser.add_argument("--num-leaves", type=num_leaves, default=100)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--repeats", type=positive_int, default=3)
    return parser.parse_args()


def main():
    args = parse_cli_args()
    result = run_benchmark(
        args.n,
        leaf_count=args.num_leaves,
        seed=args.seed,
        repeats=args.repeats,
        metric=args.metric,
    )
    print_result(result)


if __name__ == "__main__":
    main()
