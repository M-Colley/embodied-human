"""How concurrent tasks occupy the shared central stage.

Each task is a :class:`~embodied_human.response.CentralTask`: every
``period`` seconds it holds the central stage for ``duration`` seconds, and
a detection response that becomes ready meanwhile waits.

- Driving follows the ACT-R driver of threaded cognition (Salvucci and
  Taatgen, 2008, p. 120): a control update fires four rules of 50 ms, and
  while the car is stable the next update waits 500 ms. Nothing is fitted.
- A spoken n-back presents a digit every 2.25 s (Mehler et al.'s delayed
  digit recall, as used by Cooper et al., 2020). Each digit occupies the
  central stage for a chunk whose length is fitted to Conti, Dlugosch and
  Bengler (2014): ``examples/fit_drt_load.py``.
- SuRT occupies it in chunks per screen, fitted the same way; the screen
  time comes from the attentive driver.

Interleaving at chunk boundaries rather than at every 50 ms rule is the
assumption that lets a memory or search task delay responses by hundreds
of milliseconds, as observed.
"""

from __future__ import annotations

from ..core import Source
from ..response import CentralTask

#: Driving: 4 rules x 50 ms per update, plus a 500 ms pause when stable.
DRIVING = CentralTask("driving", period=0.7, duration=0.2)

#: Seconds between n-back digits.
N_BACK_PERIOD = 2.25

#: Central chunk per n-back digit, s (fitted; see the module notes).
N_BACK_CHUNK = {0: 0.743, 2: 1.115}

#: Central chunk per SuRT screen, s (fitted), and the mean screen time, s.
SURT_CHUNK = 1.771
SURT_SCREEN = 4.31

SOURCES = {
    "DRIVING": Source(
        "Salvucci & Taatgen (2008), p. 120",
        "https://doi.org/10.1037/0033-295X.115.1.101",
        "derived",
        "four 50 ms rules per control update; 500 ms delay while stable",
    ),
    "N_BACK_PERIOD": Source(
        "Cooper et al. (2020), Procedure",
        "https://doi.org/10.3389/fpsyg.2020.01154",
        "standard",
        "one digit every 2.25 s; assumed for Conti et al. (2014) as well",
    ),
    "N_BACK_CHUNK": Source(
        "Conti, Dlugosch & Bengler (2014), Table 3; fitted here",
        "https://www.hfes-europe.org/wp-content/uploads/2014/06/Conti.pdf",
        "fitted",
        "visual DRT +135 ms (0-back) and +292 ms (2-back) over driving",
    ),
    "SURT_CHUNK": Source(
        "Conti, Dlugosch & Bengler (2014), Table 3; fitted here",
        "https://www.hfes-europe.org/wp-content/uploads/2014/06/Conti.pdf",
        "fitted",
        "visual DRT +385 ms with easy SuRT over driving",
    ),
    "SURT_SCREEN": Source(
        "embodied-human attentive driver", None, "derived", "mean SuRT screen time"
    ),
}


def n_back(level: int, chunk: float | None = None) -> CentralTask:
    """Central demand of an n-back task at ``level``."""
    duration = N_BACK_CHUNK[level] if chunk is None else chunk
    return CentralTask(f"{level}-back", N_BACK_PERIOD, duration)


def surt(chunk: float | None = None, screen: float = SURT_SCREEN) -> CentralTask:
    """Central demand of continuous SuRT."""
    return CentralTask("surt", screen, SURT_CHUNK if chunk is None else chunk)
