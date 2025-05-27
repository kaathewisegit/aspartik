from ..b3 import Prior, Tree, Parameter

class Yule(Prior):
    """Hello there"""

    def __init__(self, tree: Tree, birth_rate: Parameter): ...
