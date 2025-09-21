from aspartik.rng import RNG


def random_float(lower: float, upper: float, num: int = 100) -> list[float]:
    rng = RNG(4)
    return [rng.random_float() for _ in range(num)]


def random_integer(lower: int, upper: int, num: int = 100) -> list[int]:
    rng = RNG(4)
    return [rng.random_int(lower, upper) for _ in range(num)]
