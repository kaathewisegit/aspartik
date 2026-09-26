import argparse
from time import perf_counter
from typing import Literal, get_args

from aspartik.data.tree import (
    BinaryTree,
    branch_score_matrix,
    robinson_foulds_matrix,
    triplet_distance_matrix,
)
from aspartik.rng import RNG

type Metric = Literal["robinson-foulds", "branch-score", "triplet"]


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


def distance_matrix(metric, trees):
    match metric:
        case "robinson-foulds":
            return robinson_foulds_matrix(trees)
        case "branch-score":
            return branch_score_matrix(trees)
        case "triplet":
            return triplet_distance_matrix(trees)
        case _:
            raise ValueError(f"unknown distance metric: {metric}")


def random_trees(
    metric: Metric, tree_count: int, leaf_count: int, seed: int
) -> list[BinaryTree]:
    rng = RNG(seed)
    if tree_count < 1:
        raise ValueError("expected at least one tree")
    if leaf_count < 2:
        raise ValueError("expected at least two leaves")
    return [BinaryTree.random(leaf_count, rng) for _ in range(tree_count)]


def run_benchmark(tree_count, leaf_count: int, seed: int, metric: Metric) -> float:
    generation = perf_counter()
    trees = random_trees(metric, tree_count, leaf_count, seed)
    start = perf_counter()
    print(f"generation: {start - generation:.2f}sec")
    _ = distance_matrix(metric, trees)
    end = perf_counter()

    return end - start


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=positive_int)
    parser.add_argument(
        "--metric", choices=get_args(Metric.__value__), default="robinson-foulds"
    )
    parser.add_argument("--num-leaves", type=num_leaves, default=100)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_cli_args()
    result = run_benchmark(
        args.n, leaf_count=args.num_leaves, seed=args.seed, metric=args.metric
    )
    print(f"{result}sec")


if __name__ == "__main__":
    main()
