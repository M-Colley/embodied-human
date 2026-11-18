"""Acquire a verified official TGSIM dataset using its provider metadata."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--counts", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = args.out.resolve()
raw = root / "raw"
raw.mkdir(parents=True, exist_ok=True)
meta = json.loads(args.metadata.read_text(encoding="utf-8"))
counts = json.loads(args.counts.read_text(encoding="utf-8"))
columns = [c["fieldName"] for c in meta["columns"] if not c["fieldName"].startswith(":")]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(job):
    name, url, expected = job
    path = raw / name
    for attempt in range(3):
        try:
            if not path.exists():
                with urllib.request.urlopen(url, timeout=180) as response, path.with_suffix(path.suffix + ".partial").open("wb") as target:
                    while block := response.read(1024 * 1024):
                        target.write(block)
                os.replace(path.with_suffix(path.suffix + ".partial"), path)
            rows = None
            if expected is not None:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    assert next(reader) == columns
                    rows = sum(1 for _ in reader)
                assert rows == expected, (name, rows, expected)
            record = {"file": str(path), "url": url, "bytes": path.stat().st_size, "sha256": sha256(path), "rows": rows}
            (raw / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"event": "acquired", "file": name, "rows": rows, "bytes": record["bytes"]}), flush=True)
            return record
        except Exception as error:
            print(json.dumps({"event": "retry", "file": name, "attempt": attempt + 1, "error": str(error)}), flush=True)
            if attempt == 2:
                raise


jobs = []
for row in counts:
    run = int(row["run_index"])
    total = int(row["rows"])
    for offset in range(0, total, 500000):
        query = {"$select": ",".join(columns), "$where": f"run_index={run}", "$order": "id,time,:id", "$limit": "500000", "$offset": str(offset)}
        jobs.append((f"run-{run:02d}-offset-{offset:07d}.csv", f"https://data.transportation.gov/resource/{args.dataset}.csv?" + urllib.parse.urlencode(query), min(500000, total - offset)))
data_jobs = len(jobs)
for attachment in meta.get("metadata", {}).get("attachments", []):
    name = attachment["filename"]
    if name.endswith(".csv") or name.endswith(".xlsx"):
        url = f"https://data.transportation.gov/api/views/{args.dataset}/files/{attachment['assetId']}?" + urllib.parse.urlencode({"download": "true", "filename": name})
        jobs.append((attachment["assetId"] + "-" + name, url, None))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    records = list(executor.map(fetch, jobs))
output = root / "all-runs.csv"
with output.open("wb") as target:
    for index, record in enumerate(records[:data_jobs]):
        with Path(record["file"]).open("rb") as stream:
            header = stream.readline()
            if index == 0:
                target.write(header)
            while block := stream.read(1024 * 1024):
                target.write(block)
manifest = {"status": "acquired", "source": f"https://data.transportation.gov/d/{args.dataset}", "resource": args.dataset, "rows": sum(int(row["rows"]) for row in counts), "file": str(output), "file_sha256": sha256(output), "columns": columns, "files": records, "metadata_file": str(args.metadata.resolve()), "metadata_sha256": sha256(args.metadata), "attribution": meta["attribution"], "downloaded_utc": datetime.now(timezone.utc).isoformat()}
(root / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "complete", "rows": manifest["rows"], "file": str(output), "file_sha256": manifest["file_sha256"]}), flush=True)
