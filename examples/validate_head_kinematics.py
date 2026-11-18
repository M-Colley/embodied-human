import json
import subprocess
import types
from pathlib import Path

import embodied_human as eh
from embodied_human.agents.body import BodySensing
from embodied_human.bench.integrated import surt_glances
from embodied_human.bench.protocol import benchmark_seed

baseline = "d4eaf219d453f5e2ea518414b4645a9807db1fac"
source = subprocess.check_output(
    ["git", "show", baseline + ":src/embodied_human/agents/body.py"], text=True
)
old = types.ModuleType("baseline_body")
exec(compile(source, "baseline_body.py", "exec"), old.__dict__)
corrected = BodySensing.head_motion
results = {}
try:
    for label, implementation in [
        ("before", old.BodySensing.head_motion),
        ("after", corrected),
        ("replay", corrected),
    ]:
        BodySensing.head_motion = implementation
        with benchmark_seed(0):
            results[label] = surt_glances(eh.build_human(seed=0)).as_dict()
        print(
            label,
            results[label]["metrics"],
            "motion_sickness_pct=",
            results[label]["details"]["motion_sickness_pct"],
            flush=True,
        )
finally:
    BodySensing.head_motion = corrected
report = {
    "baseline_body_revision": baseline,
    "seed": 0,
    "duration_s": 60.0,
    "results": results,
    "corrected_replay_equal": results["after"] == results["replay"],
}
output = Path("artifacts/rotating-head-benchmark.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
print("Corrected replay equal:", report["corrected_replay_equal"])
