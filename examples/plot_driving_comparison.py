"""Plot every paired recorded leader replay from an empirical campaign report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def plot_report(report: dict[str, Any], output: Path) -> int:
    """Export observed, baseline and fitted speed and gap for all paired trials."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    evaluation = report.get("evaluation", report)
    closed = evaluation.get("closed_loop", evaluation)
    pairs: dict[tuple[str, int], dict[str, Any]] = {}
    for row in closed.get("episodes", []):
        key = (row["source_group"], row["trial_index"])
        pairs.setdefault(key, {})[row["profile"]] = row
    if not pairs or any(set(pair) != {"baseline", "fitted"} for pair in pairs.values()):
        raise ValueError("complete paired baseline and fitted episodes are required")
    figure, axes = plt.subplots(
        len(pairs),
        2,
        figsize=(12, max(4, 2.05 * len(pairs))),
        squeeze=False,
        constrained_layout=True,
    )
    styles = (("Observed", "#222222"), ("Baseline", "#2874a6"), ("Fitted", "#c05621"))
    for index, (key, pair) in enumerate(sorted(pairs.items())):
        baseline, fitted = pair["baseline"], pair["fitted"]
        for column, (name, unit) in enumerate((("speed", "m/s"), ("gap", "m"))):
            axis = axes[index, column]
            suffix = "mps" if name == "speed" else "m"
            signals = (
                (baseline["time_s"], baseline[f"observed_{name}_{suffix}"]),
                (baseline["time_s"], baseline[f"predicted_{name}_{suffix}"]),
                (fitted["time_s"], fitted[f"predicted_{name}_{suffix}"]),
            )
            for (label, colour), (time, values) in zip(styles, signals):
                t, y = np.asarray(time, float), np.asarray(values, float)
                if len(t) != len(y):
                    raise ValueError("time and measurement lengths differ")
                valid = np.isfinite(t) & np.isfinite(y)
                axis.plot(t[valid], y[valid], label=label, color=colour, linewidth=1.5)
            axis.set_ylabel(f"{name.capitalize()} ({unit})")
            axis.grid(alpha=0.2)
            axis.set_title(f"{key[0]}  trial {key[1] + 1}", fontsize=10, loc="left")
            if index == len(pairs) - 1:
                axis.set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        title="Held out NGSIM replays through the MyoSuite body",
        title_fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = plot_report(json.loads(args.report.read_text()), args.out)
    print(f"Exported {count} paired trials to {args.out}")


if __name__ == "__main__":
    main()
