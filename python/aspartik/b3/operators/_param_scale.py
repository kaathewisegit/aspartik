from math import log
from typing import Literal

from ._util import sample_range
from .. import Proposal, Parameter
from ...rng import Rng


class ParamScale:
    """An operator which scales one parameter.

    This operator is analogous to BEAST2's `ScaleOperator`, except it only
    works for parameters.
    """

    # TODO: upper/lower?
    def __init__(
        self,
        param: Parameter,
        factor: float,
        distribution,
        rng: Rng,
        dimensions: Literal["one", "all", "independent"] = "all",
        weight: float = 1,
    ):
        """
        Args:
            param: The parameter to scale.
            factor:
                The scale ratio will be sampled from `(factor, 1 / factor)`.
                So, the smaller the factor, the larger the moves proposed by
                this operator are.  Also, this means that `factor` must be
                within `(0, 1)`.
            distribution:
                The distribution from which to sample the scaling factor.
            dimensions:
                Defines how multidimensional parameters will be scaled:

                - `one`: Only one dimension is scaled.
                - `all` *(default)*: All dimension are changed with the same
                    scale.
                - `independent`: All dimensions are scaled, but a new factor is
                    sampled for each of them.
        """

        if not 0 < factor < 1:
            raise ValueError(f"factor must be between 0 and 1, got {factor}")
        self.param = param
        self.factor = factor
        self.distribution = distribution
        self.rng = rng
        self.dimensions = dimensions
        self.weight = weight

    def propose(self) -> Proposal:
        low, high = self.factor, 1 / self.factor
        scale = sample_range(low, high, self.distribution, self.rng)

        match self.dimensions:
            case "one":
                index = self.rng.random_int(0, len(self.param))
                if self.param[index] == 0:
                    return Proposal.Reject()
                self.param[index] *= scale

                ratio = -log(scale)
                return Proposal.Hastings(ratio)
            case "all":
                # TODO: overload arithmetic for the whole parameter
                num_scaled = 0
                for i in range(len(self.param)):
                    if self.param[i] != 0:
                        self.param[i] *= scale
                        num_scaled += 1

                # XXX: BEAST2 claims that the Hastings ratio is (num_dimensions
                # - 1) bigger than the 1-parameter case.  The proof should be
                # in a certain Alexei/Nicholes article.  I'll have to
                # investigate, as it's unclear what's supposed to happen when
                # there are only two dimensions (or only two-non zero values).
                ratio = num_scaled * log(scale)
                return Proposal.Hastings(ratio)
            case "independent":
                ratio = 0

                for i in range(len(self.param)):
                    scale = sample_range(low, high, self.distribution, self.rng)
                    self.param[i] *= scale
                    ratio -= log(scale)

                return Proposal.Hastings(ratio)

        raise ValueError(
            f"Invalid dimensions argument.  Expected 'one', 'all', or 'literal', got {self.dimensions}"
        )
