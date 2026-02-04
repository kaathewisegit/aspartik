from typing import Self

from aspartik.stats.distributions import (
    Continuous,
    ContinuousCDF,
    Discrete,
    DiscreteCDF,
    Distribution,
    Sample,
)

# Concrete distributions

class Beta(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, shape_a: float, shape_b: float): ...
    @property
    def shape_a(self) -> float: ...
    @property
    def shape_b(self) -> float: ...

class BetaError:
    InvalidAlpha: Self
    InvalidBeta: Self

class Exp(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, rate: float): ...
    @property
    def rate(self) -> float: ...

class ExpError:
    RateInvalid: Self

class Gamma(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, shape: float, rate: float): ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...

class GammaError:
    ShapeInvalid: Self
    RateInvalid: Self
    ShapeAndRateInfinite: Self

class InverseGamma(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, shape: float, rate: float): ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...

class InverseGammaError:
    ShapeInvalid: Self
    RateInvalid: Self

class Laplace(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, location: float, scale: float): ...
    @property
    def location(self) -> float: ...
    @property
    def scale(self) -> float: ...

class LaplaceError:
    LocationInvalid: Self
    ScaleInvalid: Self

class LogNormal(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, location: float, scale: float): ...
    @property
    def location(self) -> float: ...
    @property
    def scale(self) -> float: ...

class LogNormalError:
    LocationInvalid: Self
    ScaleInvalid: Self

class Normal(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, mean: float, std_dev: float): ...

class NormalError:
    MeanInvalid: Self
    StandardDeviationInvalid: Self

class Poisson(DiscreteCDF, Discrete, Distribution, Sample[int]):
    def __init__(self, lambda_: float): ...
    @property
    def lambda_(self) -> float: ...

class PoissonError:
    LambdaInvalid: Self

class Uniform(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, min: float, max: float): ...
    @property
    def min(self) -> float: ...
    @property
    def max(self) -> float: ...

class UniformError:
    MinInvalid: Self
    MaxInvalid: Self
    MaxNotGreaterThanMin: Self
