"""A modular simulated human: senses (vision, eyes, hearing, touch, the
vestibular sense, proprioception), attention, memory, state estimation,
responses, motor control, fatigue, emotion, perceived risk and trust, as
replaceable modules with cited parameters, plus tasks and a benchmark.
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

__version__ = "0.1.1"

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
