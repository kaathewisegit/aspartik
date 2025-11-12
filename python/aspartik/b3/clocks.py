from dataclasses import dataclass
from typing import Protocol, SupportsFloat


class Clock(Protocol):
    """
    A clock model

    Currently this class only supports strict clock models.
    """

    def get_rate(self) -> float:
        """
        The uniform clock rate for all edges in the tree
        """
        ...


@dataclass(slots=True)
class StrictClock(Clock):
    """Clock model which just returns a parameter"""

    mu: SupportsFloat

    def get_rate(self) -> float:
        return float(self.mu)
