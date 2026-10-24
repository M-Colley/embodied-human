"""A modular simulated human: vision, eyes, attention, memory,
proprioception, motor control and state estimation, as replaceable modules
with cited parameters, plus secondary tasks and a benchmark.
"""

from .core import (
    CATEGORIES,
    REGISTRY,
    Human,
    HumanConfig,
    Module,
    ModuleChoice,
    Parameters,
    Source,
    available,
    build_human,
    create,
    register,
)

__version__ = "0.1.0"

__all__ = [
    "CATEGORIES",
    "REGISTRY",
    "Human",
    "HumanConfig",
    "Module",
    "ModuleChoice",
    "Parameters",
    "Source",
    "available",
    "build_human",
    "create",
    "register",
]
