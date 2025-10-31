from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import ClassVar, Protocol, SupportsFloat

from .._aspartik_rust_impl import _b3_rust_impl
from ..math import is_close

JC = _b3_rust_impl.JC
K80 = _b3_rust_impl.K80
HKY = _b3_rust_impl.HKY
