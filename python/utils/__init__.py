from collections.abc import Callable

from aspartik.b3 import Operator, Tree
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
        operator.propose()
        tree.validate()
        tree.accept()
