import os
import sys

# TODO: a temporary ugly hack
dir = os.path.abspath(__file__ + "/../../../../target/release/")
sys.path.append(dir)

from librng import Rng  # noqa: E402

__all__ = ["Rng"]


def __dir__():
    return __all__
