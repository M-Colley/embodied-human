"""Trust in automation and the decision to rely on it.

Baseline :class:`DecisionFieldTrust` is the extended decision field theory
of Gao and Lee (2006). At each decision step the person updates a belief
about the automation's capability and about their own (manual)
capability, from what they could observe:

    BC(n) = BC(n-1) + (C(n-1) - BC(n-1)) / b1       capability observed
    BC(n) = BC(n-1) + (BC_0 - BC(n-1)) / b0         not observed

Trust and self-confidence follow these beliefs with inertia ``s``:

    T(n)  = (1 - s) T(n-1)  + s BCA(n) + e
    SC(n) = (1 - s) SC(n-1) + s BCM(n) + e

and the preference ``P = T - SC`` drives reliance with hysteresis: a
person relying on automation takes over when ``P < -theta`` and hands
back when ``P > theta``. Whether a capability can be observed depends on
the automation: information-acquisition automation reveals both
capabilities all the time, action automation reveals manual capability
only while driving manually (Gao and Lee's mode-dependent case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from .core import Module, Parameters, Source, register

Mode = Literal["automation", "manual"]

_GAO_LEE = (
    "Gao & Lee (2006), Table 1",
    "https://doi.org/10.1109/TSMCA.2005.855783",
)


@dataclass(frozen=True)
class TrustParameters(Parameters):
    """Extended decision field theory.

    Args:
        opacity: b1 >= 1; how slowly beliefs follow observed capability
            (1 is a perfectly transparent interface).
        prior_pull: b0 >= 1; how slowly beliefs return to their initial
            value without information.
        noise_sd: SD of the noise in trust and self-confidence.
        inertia: s in [0, 1]; the growth-decay rate of trust.
        threshold: theta; the inhibitory threshold for switching.
        automation_belief: Initial belief in the automation's capability.
        manual_belief: Initial belief in one's own capability.
    """

    opacity: float = 2.0
    prior_pull: float = 50.0
    noise_sd: float = 0.1
    inertia: float = 0.5
    threshold: float = 0.27
    automation_belief: float = 0.744
    manual_belief: float = 0.7

    SOURCES: ClassVar = {
        "opacity": Source(*_GAO_LEE, "fitted", "2 with sound feedback, 10 with noise"),
        "prior_pull": Source(*_GAO_LEE, "fitted"),
        "noise_sd": Source(
            *_GAO_LEE,
            "derived",
            "the text gives sigma^2 = 0.1, but its theta = 1.9 sqrt(2 sigma^2) "
            "= 0.27 holds only for sigma = 0.1",
        ),
        "inertia": Source(
            *_GAO_LEE, "fitted", "taken from Lee & Moray (1992) by Gao & Lee"
        ),
        "threshold": Source(*_GAO_LEE, "fitted", "1.9 sqrt(2 sigma^2)"),
        "automation_belief": Source(
            "Hergeth, Lorenz, Vilimek & Krems (2016), Results",
            "https://doi.org/10.1177/0018720815625744",
            "measured",
            "self-reported trust of 35 drivers new to highly automated driving "
            "at its first use, 74.43% (SD 17.56), on trust's 0-1 scale",
        ),
        "manual_belief": Source("model choice", None, "assumed"),
    }

    def __post_init__(self) -> None:
        if self.opacity < 1 or self.prior_pull < 1:
            raise ValueError("opacity and prior_pull must be at least 1")
        if not 0 <= self.inertia <= 1:
            raise ValueError("inertia must lie in [0, 1]")


class Trust(Module):
    """Category ``trust``: rely on automation or not."""

    category = "trust"

    def update(
        self,
        automation_capability: float | None,
        manual_capability: float | None,
    ) -> Mode:
        """One decision step with the capabilities observed in the last
        step (``None`` when unobservable); return the mode relied on."""
        raise NotImplementedError


@register("trust", "decision_field")
class DecisionFieldTrust(Trust):
    """Gao and Lee's (2006) extended decision field theory."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = TrustParameters(**params)
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        p = self.params
        self.belief_automation = p.automation_belief
        self.belief_manual = p.manual_belief
        self.trust = p.automation_belief
        self.self_confidence = p.manual_belief
        self.mode: Mode = "automation"
        self.history: list[tuple[float, float, Mode]] = []

    @property
    def preference(self) -> float:
        """Trust minus self-confidence."""
        return self.trust - self.self_confidence

    def _belief(self, belief: float, observed: float | None, prior: float) -> float:
        p = self.params
        if observed is None:
            return belief + (prior - belief) / p.prior_pull
        return belief + (observed - belief) / p.opacity

    def update(
        self,
        automation_capability: float | None,
        manual_capability: float | None,
    ) -> Mode:
        p = self.params
        self.belief_automation = self._belief(
            self.belief_automation, automation_capability, p.automation_belief
        )
        self.belief_manual = self._belief(
            self.belief_manual, manual_capability, p.manual_belief
        )
        noise = p.noise_sd * self.rng.standard_normal(2)
        self.trust = (1 - p.inertia) * self.trust + p.inertia * self.belief_automation
        self.trust += float(noise[0])
        self.self_confidence = (
            (1 - p.inertia) * self.self_confidence
            + p.inertia * self.belief_manual
            + float(noise[1])
        )
        if self.mode == "automation" and self.preference < -p.threshold:
            self.mode = "manual"
        elif self.mode == "manual" and self.preference > p.threshold:
            self.mode = "automation"
        self.history.append((self.trust, self.self_confidence, self.mode))
        return self.mode
