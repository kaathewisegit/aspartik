from ..b3 import Prior, Tree, Parameter

class Yule(Prior):
    """Uncalibrated Yule birth-rate model"""

    def __init__(self, tree: Tree, birth_rate: Parameter): ...

class ConstantPopulation(Prior):
    """Constant population coalescent"""

    def __init__(self, tree: Tree, population: Parameter): ...
