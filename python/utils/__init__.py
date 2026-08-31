from collections.abc import Callable

from aspartik.b3 import Operator, Proposal
from aspartik.b3.parameters import Tree
from aspartik.data.msa import MSA
from aspartik.rng import RNG


def random_float(lower: float, upper: float, num: int = 100) -> list[float]:
    rng = RNG(4)
    return [rng.random_float(lower, upper) for _ in range(num)]


def random_integer(lower: int, upper: int, num: int = 100) -> list[int]:
    rng = RNG(4)
    return [rng.random_int(lower, upper) for _ in range(num)]


def check_tree_operator(factory: Callable[[Tree], Operator]) -> None:
    rng = RNG(4)
    tree = Tree([str(i) for i in range(100)], rng)

    operator = factory(tree)

    for _ in range(1000):
        proposal = operator.propose()
        if proposal == Proposal.Abort():
            tree.reject()
        else:
            tree.accept()

        tree.validate()


def random_tree(rng, lower: int, upper: int):
    len = rng.random_int(lower, upper)
    return Tree([str(i) for i in range(len)], rng)


def random_trees(lower: int, upper: int, num: int = 1000):
    rng = RNG(4)

    return [random_tree(rng, lower, upper) for _ in range(num)]


def random_msa(rng: RNG, lower: int, upper: int):
    num_sites = rng.random_int(lower, upper)
    num_sequences = rng.random_int(lower, upper)

    return MSA.random(
        num_sequences, num_sites, [str(i) for i in range(num_sequences)], rng
    )


def random_msas(lower: int, upper: int, num: int):
    rng = RNG(4)

    return [random_msa(rng, lower, upper) for _ in range(num)]
