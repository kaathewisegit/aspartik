from typing import List

from .. import Proposal, Parameter
from ...rng import Rng


class DeltaExchange:
    """Operator which tweaks a multidimensional parameter without changing its sum.

    This operator is analogous to BEAST2's `DeltaExchangeOperator`.  It picks
    two random dimensions from a set list of parameters, a random delta, and
    increments one of them by delta and decrements the other one.  The ratio of
    the decrement can be controlled with the `weights` vector: the value of the
    decrement is `delta * weights[inc_param] / weights[dec_param]`.
    """

    def __init__(
        self,
        params: List[Parameter],
        weights: List[float],
        factor: float,
        rng: Rng,
        weight: float = 1,
    ):
        """
        Args:
            params:
                A list of parameters to edit.  Two random ones will be sampled
                for each proposal.
            weights:
                The weights which define the sum relations between parameters.
                Must have the same length as the `params` list.
            factor:
                The move size is a random value between 0 and 1 multiplied by
                `factor`.
        """

        if len(params) != len(weights):
            raise ValueError(
                f"Length of `params` and `weight` must be the same.  Got {len(params)} and {len(weights)}"
            )

        self.params = params
        self.weights = weights
        self.factor = factor
        self.weight = weight
        self.rng = rng

        self.dimensions = []
        for param_i, param in enumerate(self.params):
            for dim_i in range(len(param)):
                self.dimensions.append((param_i, dim_i))

        self.num_dimensions = 0
        for param in self.params:
            self.num_dimensions += len(param)

    def propose(self) -> Proposal:
        # TODO: zero weights

        delta = self.rng.random_float() * self.factor

        dim_1 = self.rng.random_int(0, len(self.dimensions))
        dim_2 = self.rng.random_int(0, len(self.dimensions) - 1)
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
