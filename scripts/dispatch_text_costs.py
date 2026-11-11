"""After the existing audit, enqueue the GPU benchmark behind any newly submitted retries."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dependency_jobs(queue_lines, own_job):
    jobs = set()
    for line in queue_lines:
        job = line.strip()
        if not re.fullmatch(r"\d+(?:_\d+)?", job):
            raise ValueError(f"Unexpected Slurm job ID: {job!r}")
        parent = job.split("_")[0]
        if parent != str(own_job):
            jobs.add(parent)
    return sorted(jobs, key=int)


def main():
    own_job = os.environ["SLURM_JOB_ID"]
    report = ROOT / ".log" / f"text_costs_dispatch_{own_job}.json"
    if report.exists():
        raise RuntimeError(f"Dispatch record already exists: {report}")
    queue = subprocess.run(["squeue", "--me", "--array", "-h", "-o", "%i"], check=True, capture_output=True, text=True)
    jobs = dependency_jobs(queue.stdout.splitlines(), own_job)
    command = ["sbatch", "--parsable"]
    if jobs:
        command.append("--dependency=afterany:" + ":".join(jobs))
    command.append(str(ROOT / "scripts" / "run_text_costs.sbatch"))
    submitted = subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
    job_id = submitted.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {submitted.stdout}")
    record = {
        "dispatcher_job_id": own_job,
        "benchmark_job_id": job_id,
        "afterany_jobs": jobs,
        "command": command,
        "output": str(ROOT / "results" / "text_costs" / f"job_{job_id}"),
    }
    report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
