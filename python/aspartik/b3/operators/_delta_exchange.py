from typing import List

from ._util import sample_range
from .. import State, Proposal, Parameter


class DeltaExchange:
    def __init__(
        self,
        params: List[Parameter],
        weights: List[float],
        delta: float,
        distribution,
        weight: float = 1,
    ):
        if len(params) != len(weights):
            raise ValueError(
                f"Length of `params` and `weight` must be the same.  Got {len(params)} and {len(weights)}"
            )

        self.params = params
        self.weights = weights
        self.delta = delta
        self.distribution = distribution
        self.weight = weight

        self.dimensions = []
        for param_i, param in enumerate(self.params):
            for dim_i in range(len(param)):
                self.dimensions.append((param_i, dim_i))

        self.num_dimensions = 0
        for param in self.params:
            self.num_dimensions += len(param)

    def propose(self, state: State) -> Proposal:
        # TODO: zero weights

        rng = state.rng

        low, high = self.delta, 1 / self.delta
        delta = sample_range(low, high, self.distribution, rng)

        dim_1 = rng.random_int(0, len(self.dimensions))
        dim_2 = rng.random_int(0, len(self.dimensions) - 1)
        # dim_1 and dim_2 must be different.
        if dim_1 == dim_2:
            # If we hit the same dimension, we increment the first one.  We can
            # do the increment safely because if dim_1 is the last one then it
            # doesn't equal dim_2
            dim_1 += 1

        (param_1, dim_1) = self.dimensions[dim_1]
        (param_2, dim_2) = self.dimensions[dim_2]

        self.params[param_1][dim_1] -= delta
        self.params[param_2][dim_2] += delta * (
            self.weights[param_1] / self.weights[param_2]
        )

        # The move is symmetrical, so the Hastings ratio is 0
        return Proposal.Hastings(0)
