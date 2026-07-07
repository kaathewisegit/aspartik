from typing import Self, SupportsFloat

from aspartik.stats.distributions import (
    Continuous,
    ContinuousCDF,
    Distribution,
    Sample,
)

# Concrete distributions

class Beta(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, shape_a: SupportsFloat, shape_b: SupportsFloat): ...
    @property
    def shape_a(self) -> float: ...
    @property
    def shape_b(self) -> float: ...

class BetaError:
    InvalidAlpha: Self
    InvalidBeta: Self

class Exp(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, rate: SupportsFloat): ...
    @property
    def rate(self) -> float: ...

class Gamma(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, shape: SupportsFloat, rate: SupportsFloat): ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...

class GammaError:
    ShapeInvalid: Self
    RateInvalid: Self
    ShapeAndRateInfinite: Self

class InverseGamma(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, shape: SupportsFloat, rate: SupportsFloat): ...
    @property
    def shape(self) -> float: ...
    @property
    def rate(self) -> float: ...

class InverseGammaError:
    ShapeInvalid: Self
    RateInvalid: Self

class Laplace(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, location: SupportsFloat, scale: SupportsFloat): ...
    @property
    def location(self) -> float: ...
    @property
    def scale(self) -> float: ...

class LaplaceError:
    LocationInvalid: Self
    ScaleInvalid: Self

class LogNormal(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, location: SupportsFloat, scale: SupportsFloat): ...
    @property
    def location(self) -> float: ...
    @property
    def scale(self) -> float: ...

class LogNormalError:
    LocationInvalid: Self
    ScaleInvalid: Self

class Normal(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, mean: SupportsFloat, std_dev: SupportsFloat): ...

class NormalError:
    MeanInvalid: Self
    StandardDeviationInvalid: Self

class Uniform(ContinuousCDF, Continuous, Distribution, Sample[float]):
    def __init__(self, min: SupportsFloat, max: SupportsFloat): ...
    @property
    def min(self) -> float: ...
    @property
    def max(self) -> float: ...

class UniformError:
    MinInvalid: Self
    MaxInvalid: Self
    MaxNotGreaterThanMin: Self
