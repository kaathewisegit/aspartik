from aspartik.rng import RNG


def random_float(lower, upper, num=100):
    rng = RNG(4)
    return [rng.random_float() for _ in range(num)]
