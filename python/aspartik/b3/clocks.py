from dataclasses import dataclass

from . import Parameter


@dataclass
class StrictClock:
    """Clock model which just returns a parameter"""

    mu: Parameter

    def get_rate(self):
        return self.mu[0]
