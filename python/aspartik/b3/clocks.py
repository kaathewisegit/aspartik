from dataclasses import dataclass
from typing import SupportsFloat


@dataclass
class StrictClock:
    """Clock model which just returns a parameter"""

    mu: SupportsFloat

    def get_rate(self):
        return float(self.mu)
