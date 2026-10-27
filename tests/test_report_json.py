"""Reports must be usable by strict JSON clients, including a browser."""

import json
import math

import numpy as np

import embodied_human as eh
from embodied_human.bench.__main__ import main
from embodied_human.bench.benchmarks import BENCHMARKS, Result, json_safe, run


def test_unscored_benchmark_is_not_a_pass():
    result = Result("custom", ("memory",))
    assert not result.passed
    assert result.as_dict()["score"] is None


def test_report_replaces_nonfinite_metrics_with_null(tmp_path, monkeypatch):
    def undefined(human):
        return Result(
            "custom.undefined",
            ("memory",),
            metrics={"custom.undefined": math.inf},
            details={"ci": [math.nan, np.float64(1.0)]},
        )

    monkeypatch.setitem(BENCHMARKS, "custom.undefined", undefined)
    path = tmp_path / "report.json"
    assert main(["--only", "custom.undefined", "--out", str(path)]) == 0

    def reject(value):
        raise AssertionError(f"nonstandard JSON token: {value}")

    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    assert result["results"][0]["metrics"]["custom.undefined"] is None
    assert result["results"][0]["details"]["ci"] == [None, 1.0]


def test_explicit_empty_benchmark_selection_runs_nothing(tmp_path):
    assert run(eh.build_human(seed=0), []) == []
    path = tmp_path / "empty.json"
    assert main(["--only", "--out", str(path)]) == 0
    assert json.loads(path.read_text())["results"] == []
    assert json_safe(np.array([1.0, np.nan])) == [1.0, None]
    assert json_safe(np.array(np.nan)) is None
    assert json_safe(np.longdouble("0.5")) == 0.5
    assert json_safe(np.array(["0.5", "nan"], dtype=np.longdouble)) == [0.5, None]
