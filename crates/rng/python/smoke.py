from aspartik.rng import RNG

os = RNG()
from_seed = RNG(4)
rng = from_seed

rng.random_bool()
rng.random_bool(1 / 3)
rng.random_int(lower=10, upper=30)
