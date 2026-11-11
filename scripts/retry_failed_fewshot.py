"""One-shot audit and resubmission after the four original Slurm arrays terminate."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path('/homes/bcalderara/merge-and-rebase')
CHECKPOINT_ROOT = Path('/work/tesi_bcalderara')
PREVIOUS_CHECKPOINT_ROOTS = (Path('/work/intesasanpaolo_phd'), Path('/work/bcalde_phd'))
FAMILIES = {
    'vitb16_full_fewshot': 'run_vitb16_laion2b_full_fewshot.sbatch',
    'vitb16_laion2b_lora_fewshot': 'run_vitb16_laion2b_lora_fewshot.sbatch',
    'vitl14_datacomp_full_fewshot': 'run_vitl14_datacomp_full_fewshot.sbatch',
    'vitl14_datacomp_lora_fewshot': 'run_vitl14_datacomp_lora_fewshot.sbatch',
}
PATTERN = re.compile(r'seed_(\d+)_job_(\d+)_(\d+)_config.yaml$')


def audit(root: Path):
    records = []
    for family, script in FAMILIES.items():
        folder = root / '.log' / family
        statuses = {}
        for path in folder.glob('*/*/run_*.json'):
            try:
                run = json.loads(path.read_text())['run_logging']
                config = Path(run['metadata']['config_path']).name
                statuses.setdefault(config, []).append(run)
            except (ValueError, KeyError):
                continue
        groups = {}
        for path in folder.glob('*/*/seed_*_job_*_config.yaml'):
            match = PATTERN.fullmatch(path.name)
            if not match:
                continue
            seed, job, index = map(int, match.groups())
            groups.setdefault((path.parent.parent.name, path.parent.name, seed), []).append((job, index, path))
        for key, attempts in sorted(groups.items()):
            job, index, config = max(attempts)
            runs = statuses.get(config.name, [])
            latest = max(runs, key=lambda r: r['created_unix']) if runs else None
            status = latest['status'] if latest else 'missing_summary'
            records.append(dict(family=family, script=script, task=key[0], shots=key[1], seed=key[2],
                                previous_job=job, index=index, config=str(config), status=status))
    return records


def retry_script(record, directory):
    """Keep resource directives; run the exact saved experiment rather than today's task mapping."""
    source = (ROOT / 'scripts' / record['script']).read_text()
    headers = [s for s in source.splitlines() if s.startswith('#SBATCH') and '--array=' not in s]
    cfg = yaml.safe_load(Path(record['config']).read_text())
    assert cfg['datasets_order'] == [record['task']], record
    assert cfg['common']['seed'] == record['seed'], record
    old_output = Path(cfg['common']['output']['out_dir'])
    for root in (CHECKPOINT_ROOT, *PREVIOUS_CHECKPOINT_ROOTS):
        if old_output.is_relative_to(root):
            cfg['common']['output']['out_dir'] = str(CHECKPOINT_ROOT / old_output.relative_to(root))
            break
    else:
        raise ValueError(f'Unrecognized checkpoint root: {old_output}')
    snapshot = directory / f"{record['family']}_{record['task']}_{record['shots']}_seed{record['seed']}.yaml"
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
    folder = Path(record['config']).parent
    prefix = f"{folder}/seed_{record['seed']}_job_"
    body = f'''#!/bin/bash
{chr(10).join(headers)}
set -euo pipefail
cd {shlex.quote(str(ROOT))}
export PYTHONUNBUFFERED=1
export PYTHONPATH={shlex.quote(str(ROOT / 'src'))}
export HF_HUB_DISABLE_PROGRESS_BARS=1
export MR_SVHN_PREPROCESS=1
export MR_SVHN_TARGET_SIZE=96
export MR_SVHN_AUGMENTATION=none
# Inherit the normal HF cache, matching sbatch_granularity.sh.
PREFIX={shlex.quote(prefix)}${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
exec >"${{PREFIX}}.log" 2>&1
cp {shlex.quote(str(snapshot))} "${{PREFIX}}_config.yaml"
echo "Retry of {record['previous_job']}_{record['index']}; original config: {record['config']}"
{shlex.quote(str(ROOT / '.venv/bin/python'))} scripts/run_fewshot_with_storage_fallback.py "${{PREFIX}}_config.yaml" {shlex.quote(record['task'])} "${{PREFIX}}.log"
'''
    path = snapshot.with_suffix('.sbatch')
    path.write_text(body)
    subprocess.run(['bash', '-n', str(path)], check=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.submit:
        queue = subprocess.run(['squeue', '-h', '-o', '%u|%j'], check=True, text=True, capture_output=True)
        active = [line for line in queue.stdout.splitlines()
                  if line.split('|')[0] == 'bcalderara' and line.split('|')[-1] in FAMILIES]
        if active:
            raise RuntimeError(f'Original fine-tuning jobs are still active: {active}')
    directory = ROOT / '.log/fewshot_retry' / ('dispatch_' + time.strftime('%Y%m%d_%H%M%S'))
    directory.mkdir(parents=True, exist_ok=False)
    records = audit(ROOT)
    report = directory / 'report.json'
    for record in records:
        record['action'] = 'skip_success' if record['status'] == 'success' else 'retry'
    report.write_text(json.dumps(records, indent=2))
    for record in records:
        if record['action'] != 'retry':
            continue
        script = retry_script(record, directory)
        if args.submit:
            # Submit individually, retaining the original task index in log filenames.
            result = subprocess.run(['sbatch', '--parsable', f"--array={record['index']}", str(script)],
                                    text=True, capture_output=True)
            record['submission_stdout'] = result.stdout.strip()
            record['submission_stderr'] = result.stderr.strip()
            record['action'] = 'submitted' if result.returncode == 0 else 'submission_failed'
            report.write_text(json.dumps(records, indent=2))
        print(record['family'], record['task'], record['shots'], record['status'], record['action'], flush=True)
    print(f'Report: {report}', flush=True)
    if any(r['action'] == 'submission_failed' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
