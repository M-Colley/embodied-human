"""Native PyACTUp integration and assumed serial position probe comparisons."""

import math
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from embodied_human.memory import ActRMemory, Retrieval
from embodied_human.memory_pyactup import PyACTUpMemory
from embodied_human.tasks.recall import run_serial_recall


def test_pyactup_construction_and_focus_work_without_the_extra():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['pyactup'] = None; "
            "from embodied_human.memory_pyactup import PyACTUpMemory; "
            "m = PyACTUpMemory(); m.encode('a', [1, 2], 0.0); "
            "assert m.retrieve('a', 0.0).value == [1, 2]",
        ],
        check=True,
        capture_output=True,
    )


def test_missing_backend_has_an_actionable_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("no pyactup", name="pyactup")

    monkeypatch.setattr("embodied_human.memory_pyactup.import_module", missing)
    memory = PyACTUpMemory(capacity=0)
    memory.encode("a", 1, 0.0)
    with pytest.raises(ImportError, match=r"embodied-human\[memory\]"):
        memory.retrieve("a", 1.0)


def test_native_activation_latency_and_rehearsal_match_the_equations():
    pytest.importorskip("pyactup")
    memory = PyACTUpMemory(capacity=0, noise=0.0, threshold=-5.0)
    memory.encode("a", [1, 2], 0.0)
    memory.encode("a", [3, 4], 1.0)
    result = memory.retrieve("a", 3.0)
    activation = math.log(3.0**-0.5 + 2.0**-0.5)
    assert result.success and result.value == [3, 4]
    assert result.activation == pytest.approx(activation)
    assert result.latency == pytest.approx(math.exp(-activation))
    assert memory.base_level("a", 3.0) == pytest.approx(activation)


def test_partial_matching_and_native_blending_use_feature_content():
    pytest.importorskip("pyactup")
    memory = PyACTUpMemory(capacity=0, noise=0.0, threshold=-5.0)
    memory.encode_chunk("older", "A", 0.0, {"position": 1, "rating": 10.0})
    memory.encode_chunk("newer", "B", 1.0, {"position": 2, "rating": 20.0})
    assert not memory.retrieve_cued({"position": 3}, 2.0, partial=False).success
    memory.similarity("position", lambda x, y: max(0.0, 1.0 - abs(x - y) / 10))
    result = memory.retrieve_cued({"position": 3}, 2.0)
    assert result.key == "newer" and result.value == "B"
    assert result.activation == pytest.approx(-0.1)
    expected = (10.0 * 2.0**-0.5 + 20.0) / (2.0**-0.5 + 1.0)
    assert memory.blend("rating", {}, 2.0) == pytest.approx(expected)
    memory.reset(np.random.default_rng(1))
    memory.encode_chunk("new", "C", 0.0, {"position": 2})
    assert memory.retrieve_cued({"position": 3}, 1.0).key == "new"


def test_native_operations_preserve_global_random_streams_and_reset_repeats():
    pytest.importorskip("pyactup")
    python_before = random.getstate()
    numpy_before = np.random.get_state()

    def run():
        memory = PyACTUpMemory(capacity=0, decay=0.0, threshold=-5.0)
        memory.reset(np.random.default_rng(71))
        memory.encode_chunk("a", 1, 0.0, {"kind": "item", "rating": 1.0})
        memory.encode_chunk("b", 2, 0.0, {"kind": "item", "rating": 3.0})
        keys = [memory.retrieve_cued({"kind": "item"}, 1.0).key for _ in range(40)]
        return keys, memory.blend("rating", {"kind": "item"}, 1.0)

    assert run() == run()
    assert random.getstate() == python_before
    for before, after in zip(numpy_before, np.random.get_state()):
        np.testing.assert_equal(before, after)


def test_retrieval_ties_are_independent_across_threads():
    pytest.importorskip("pyactup")

    def run(seed):
        memory = PyACTUpMemory(capacity=0, decay=0.0, noise=0.0, threshold=-5.0)
        memory.reset(np.random.default_rng(seed))
        memory.encode_chunk("a", 1, 0.0, {"kind": "item"})
        memory.encode_chunk("b", 2, 0.0, {"kind": "item"})
        return [memory.retrieve_cued({"kind": "item"}, 1.0).key for _ in range(80)]

    expected = run(72)
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(run, [72] * 8))
    assert all(output == expected for output in outputs)
    assert {"a", "b"} == set(expected)


def test_context_interference_produces_identity_scored_intrusions():
    pytest.importorskip("pyactup")
    settings = {"capacity": 0, "noise": 0.0, "threshold": -5.0, "mismatch_penalty": 0.1}
    clean = run_serial_recall(PyACTUpMemory(**settings), [1, 2, 3]).summary()
    exact = run_serial_recall(
        PyACTUpMemory(**settings), [1, 2, 3], distractors=[[1, 2, 3]], partial=False
    ).summary()
    partial = run_serial_recall(
        PyACTUpMemory(**settings), [1, 2, 3], distractors=[[1, 2, 3]], partial=True
    ).summary()
    # Same payloads in the later list still count as intrusions by identity.
    assert clean["accuracy"] == exact["accuracy"] == 1.0
    assert partial["intrusion_rate"] == 1.0 and partial["accuracy"] == 0.0
    baseline = run_serial_recall(
        ActRMemory(capacity=0, noise=0.0, threshold=-5.0),
        [1, 2, 3],
        distractors=[[1, 2, 3]],
    ).summary()
    assert baseline["intrusion_rate"] == 0.0
    assert len(partial["correct_by_position"]) == 3


def test_serial_position_access_gradient_matches_native_decay_threshold():
    pytest.importorskip("pyactup")
    result = run_serial_recall(
        PyACTUpMemory(capacity=0, noise=0.0, threshold=-0.6), list("abcdef")
    ).summary()
    assert result["correct_by_position"] == [False, False, False, True, True, True]
    assert result["failure_rate"] == 0.5 and result["intrusion_rate"] == 0.0


def test_serial_recall_checks_successful_content_and_applies_focus_consistently():
    class WrongContentMemory(ActRMemory):
        def retrieve(self, key, time):
            return Retrieval("wrong", True, 0.1, 1.0)

    wrong = run_serial_recall(WrongContentMemory(), [1, 2, 3]).summary()
    assert wrong["accuracy"] == 0
    assert wrong["content_error_rate"] == 1
    assert wrong["intrusion_rate"] == wrong["failure_rate"] == 0
    settings = {"capacity": 6, "noise": 0.0, "threshold": 10.0}
    baseline = run_serial_recall(
        ActRMemory(**settings), [1, 2, 3], distractors=[[1, 2, 3]]
    ).summary()
    native = run_serial_recall(
        PyACTUpMemory(**settings), [1, 2, 3], distractors=[[1, 2, 3]]
    ).summary()
    assert native["accuracy"] == baseline["accuracy"] == 1
    assert native["intrusion_rate"] == 0


def test_invalid_features_and_simultaneous_declarative_queries_are_rejected():
    memory = PyACTUpMemory()
    with pytest.raises(ValueError, match="hashable"):
        memory.encode_chunk("bad", None, 0.0, {"x": ([1],)})
    memory.encode_chunk("a", None, 0.0, {"x": 1})
    with pytest.raises(ValueError, match="new key"):
        memory.encode_chunk("a", None, 1.0, {"x": 2})
    with pytest.raises(ValueError, match="follow"):
        memory.retrieve_cued({"x": 1}, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        PyACTUpMemory().encode("a", None, -1.0)
