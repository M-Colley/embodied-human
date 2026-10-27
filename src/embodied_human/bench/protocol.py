"""Random seeds for nested benchmark protocols."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

_SEED: ContextVar[int] = ContextVar("embodied_human_benchmark_seed", default=0)


def protocol_seed(index: int | None = None) -> int:
    """Derive nested streams while preserving the zero seed protocol."""
    parent = _SEED.get()
    if index is None:
        return parent
    if type(index) is not int or index < 0:
        raise ValueError("trial index must be a nonnegative integer")
    if parent == 0:
        return index
    return int(np.random.SeedSequence([parent, index, 0x4548]).generate_state(1)[0])


def protocol_seeds(count: int) -> list[int]:
    return [protocol_seed(index) for index in range(count)]


@contextmanager
def benchmark_seed(seed: int) -> Iterator[None]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 4294967295]")
    token = _SEED.set(seed)
    try:
        yield
    finally:
        _SEED.reset(token)
