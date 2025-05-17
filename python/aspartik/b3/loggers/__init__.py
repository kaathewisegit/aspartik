"""Classes which record the state of the simulation.

All classes here adhere to the `Logger` protocol and can be passed to the `run`
function.
"""

from ._tree import TreeLogger as TreeLogger

__all__ = ["TreeLogger"]
