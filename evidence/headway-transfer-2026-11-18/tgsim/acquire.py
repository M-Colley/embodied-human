"""Preserve official TGSIM run rows, geometries and dictionaries."""

import concurrent.futures
import csv
import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
META = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
COUNTS = json.loads((ROOT / "run_counts.json").read_text(encoding="utf-8"))
BASE = "https://data.transportation.gov"
COLUMNS = ["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "speed_kf", "acceleration_kf", "length_smoothed", "width_smoothed", "type_most_common", "acc", "run_index"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(job):
    name, url, expected = job
    target = RAW / name
    for attempt in range(3):
        try:
            if not target.exists():
                with urllib.request.urlopen(url, timeout=120) as response, target.with_suffix(target.suffix + ".partial").open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                os.replace(target.with_suffix(target.suffix + ".partial"), target)
            rows = None
            if expected is not None:
                with target.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    assert next(reader) == COLUMNS
                    rows = sum(1 for _ in reader)
                assert rows == expected, (name, rows, expected)
            record = {"file": str(target), "url": url, "bytes": target.stat().st_size, "sha256": digest(target), "rows": rows}
            (RAW / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps({"event": "acquired", **record}), flush=True)
            return record
        except Exception as error:
            print(json.dumps({"event": "retry", "file": name, "attempt": attempt + 1, "error": str(error)}), flush=True)
            if attempt == 2:
                raise


jobs = []
for row in COUNTS:
    run = int(row["run_index"])
    query = {"$select": ",".join(COLUMNS), "$where": f"run_index={run}", "$order": "id,time,:id", "$limit": "1000000"}
    jobs.append((f"run-{run:02d}.csv", BASE + "/resource/7zjf-a4zf.csv?" + urllib.parse.urlencode(query), int(row["rows"])))
for attachment in META["metadata"]["attachments"]:
    name = attachment["filename"]
    if name.endswith(".csv") or name.endswith(".xlsx"):
        unique = attachment["assetId"] + "-" + name
        url = BASE + f"/api/views/7zjf-a4zf/files/{attachment['assetId']}?" + urllib.parse.urlencode({"download": "true", "filename": name})
        jobs.append((unique, url, None))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch, jobs))
output = ROOT / "tgsim-i294-l1-all-runs.csv"
with output.open("wb") as target:
    for index, record in enumerate(results[:len(COUNTS)]):
        with Path(record["file"]).open("rb") as stream:
            header = stream.readline()
            if index == 0:
                target.write(header)
            while block := stream.read(1024 * 1024):
                target.write(block)
manifest = {"status": "acquired", "dataset_id": "tgsim-i294-l1", "doi": "10.21949/1404217", "source": "https://data.transportation.gov/d/7zjf-a4zf", "rows": sum(int(row["rows"]) for row in COUNTS), "file": str(output), "file_sha256": digest(output), "files": results, "columns": COLUMNS, "metadata_sha256": digest(ROOT / "metadata.json"), "attribution": META["attribution"], "rights": META["metadata"]["custom_fields"]["Common Core 3"]["Rights"], "license": META.get("license"), "downloaded_utc": datetime.now(timezone.utc).isoformat()}
(ROOT / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"event": "complete", "rows": manifest["rows"], "file": str(output), "file_sha256": manifest["file_sha256"]}), flush=True)
