from typing import Optional, List

__all__: List[str]

class Rng:
    """Random numbers generator.

    It's backed by a 64-bit output PCG, see the [Rust documentation][pcg] for
    details.

    It can be used standalone, but it was created as a Rust-native RNG which
    can be used efficiently by other Aspartik modules.

    [pcg]: https://docs.rs/rand_pcg/latest/rand_pcg/type.Pcg64.html
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed:
                A positive integer less than $2^{64}$.  If no seed is passed the
                RNG will be seeded from the operating system data source.
        """
    def random_bool(self, ratio: float = 0.5) -> bool:
        """
        Args:
            ratio:
                The probability of returning `True`.  Must be between in the
                range $[0, 1]$.
        """
    def random_ratio(self, numerator: int, denominator: int) -> bool:
        """Returns `True` with the chance of $\\frac{numerator}{denominator}$.

        Due to the FFI overhead it probably won't be faster than using
        `random_bool(numerator / denominator)`, but it might express the intent
        more clearly.
        """
    def random_int(self, lower: int, upper: int) -> int:
        """Returns a random int in $[lower, upper)$."""
    def random_float(self) -> float:
        """Returns a float uniformly distributed on $[0, 1)$

        See the [`rand` notes][float] for details on how the values are
        sampled.

        [float]: https://docs.rs/rand/latest/rand/distr/struct.StandardUniform.html#floating-point-implementation
        """
