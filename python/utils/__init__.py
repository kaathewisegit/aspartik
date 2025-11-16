from collections.abc import Callable

from aspartik.b3 import Operator, Proposal, Tree
from aspartik.rng import RNG


def random_float(lower: float, upper: float, num: int = 100) -> list[float]:
    rng = RNG(4)
    return [rng.random_float() for _ in range(num)]


def random_integer(lower: int, upper: int, num: int = 100) -> list[int]:
    rng = RNG(4)
    return [rng.random_int(lower, upper) for _ in range(num)]


def check_tree_operator(factory: Callable[[Tree], Operator]) -> None:
    rng = RNG(4)
    tree = Tree([str(i) for i in range(100)], rng)

    operator = factory(tree)

    for _ in range(1000):
        proposal = operator.propose()
        if proposal == Proposal.Reject():
            tree.reject()
        else:
            print(proposal)
            tree.accept()

        tree.validate()


type Frequencies = tuple[float, float, float, float]


def random_frequencies(rng: RNG) -> Frequencies:
    out = [rng.random_float() for _ in range(4)]
    total = sum(out)
    out = [el / total for el in out]
    out = tuple(out)
    assert len(out) == 4
    return out


def random_frequencies_many(num: int = 1000) -> list[Frequencies]:
    rng = RNG(4)

    return [random_frequencies(rng) for _ in range(num)]


def random_tree(rng, lower: int, upper: int):
    len = rng.random_int(lower, upper)
    return Tree([str(i) for i in range(len)], rng)


def random_trees(lower: int, upper: int, num: int = 1000):
    rng = RNG(4)

    return [random_tree(rng, lower, upper) for _ in range(num)]
