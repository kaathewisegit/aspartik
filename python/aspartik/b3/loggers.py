"""Classes which record the state of the simulation.

All classes here adhere to the `Callback` protocol and should be passed as
such.
"""

import time
from compression.zstd import ZstdCompressor
from dataclasses import dataclass, field
from io import BufferedWriter
from typing import Optional

from . import MCMC, Callback
from .parameters import Tree


class _LogWriter:
    _file: BufferedWriter
    _compressor: Optional[ZstdCompressor]

    __slots__ = ("_compressor", "_file")

    def __init__(self, path: str, zstd: bool = False):
        if zstd:
            self._file = open(f"{path}.zst", "wb")
            self._compressor = ZstdCompressor()
        else:
            self._file = open(path, "wb")
            self._compressor = None

    def writeln(self, line: str) -> None:
        encoded = f"{line}\n".encode()

        if self._compressor:
            compressed = self._compressor.compress(encoded)
            self._file.write(compressed)
        else:
            self._file.write(encoded)

    def flush(self) -> None:
        if self._compressor:
            compressed = self._compressor.flush()
            self._file.write(compressed)

        self._file.flush()


@dataclass(slots=True)
class TreeLogger(Callback):
    """
    Records the structure of the tree in Newick format

    The exact format in the resulting file is a collection of tree structures
    on each recorded step delimited by newlines
    """

    tree: Tree
    path: str
    """
    Path to the file to write to.  It's opened verbatim (so the `.trees`
    extension won't be added automatically).
    """
    every: int
    """How often the logger will be called"""

    zstd: bool = False
    """
    Compress the output with zstd

    If enabled, the logger will compress its output and write it to a
    `{path}.zst`.
    """

    _writer: _LogWriter = field(init=False)

    def __post_init__(self):
        self._writer = _LogWriter(self.path, zstd=self.zstd)

    def call(self, mcmc: MCMC):
        newick = self.tree.newick()
        self._writer.writeln(newick)

    def finish(self):
        self._writer.flush()

    def __getstate__(self):
        # None ignores __dict__ which contains the file handle
        return (None, self.__slots__)


@dataclass(slots=True)
class PrintLogger(Callback):
    """
    Prints the simulation progress onto the screen

    Currently it only supports the step index, posterior/likelihood/total
    prior, and speed in time per million steps.
    """

    every: int

    _last_time: Optional[float] = field(init=False, default=None)

    def call(self, mcmc: MCMC):
        if mcmc.current_step == 0:
            print(
                f"{'Step':>10}{'Posterior':>15}{'Likelihood':>15}{'Prior':>15}{'Speed t/m':>15}"
            )

        current_time = time.perf_counter()
        if self._last_time:
            # in seconds
            speed = (current_time - self._last_time) / self.every * 1_000_000

            speed = f"{speed / 60:.1f}min" if speed >= 60 else f"{speed:.0f}sec"
        else:
            speed = "-"

        likelihood = mcmc.likelihood.likelihood()
        print(
            f"{mcmc.current_step:>10}{mcmc.posterior:>15.2f}{likelihood:>15.2f}{mcmc.prior:>15.2f}{speed:>15}"
        )

        self._last_time = current_time
