from __future__ import annotations

from collections.abc import Sequence

class Logger:
    def __init__(self): ...
    def with_targets(self, targets: Sequence[str]) -> Logger: ...
    def with_level(self, level: Level) -> Logger: ...
    def to_file(self, path: str) -> Logger: ...
    def init(self) -> None: ...

class Level:
    Trace: Level
    Debug: Level
    Info: Level
    Warn: Level
    Error: Level
