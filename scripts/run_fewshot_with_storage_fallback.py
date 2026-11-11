"""Run a saved fine-tuning config, redirecting checkpoint storage on quota errors."""
import errno
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PRIMARY = Path('/work/intesasanpaolo_phd')
FALLBACK = Path('/work/bcalde_phd')


def redirect(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    old = Path(cfg['common']['output']['out_dir'])
    if not old.is_relative_to(PRIMARY):
        return False
    new = FALLBACK / old.relative_to(PRIMARY)
    new.mkdir(parents=True, exist_ok=True)
    cfg['common']['output']['out_dir'] = str(new)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f'Checkpoint storage fallback: {old} -> {new}', flush=True)
    return True


def preflight(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    out = Path(cfg['common']['output']['out_dir'])
    try:
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=out) as probe:
            probe.write(b'checkpoint storage probe\n')
            probe.flush()
    except OSError as exc:
        if exc.errno not in (errno.EDQUOT, errno.ENOSPC) or not redirect(config_path):
            raise


def main():
    config_path, task, log = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    preflight(config_path)
    command = [sys.executable, '-m', 'merge_and_rebase.finetune.train_vision',
               '--vision-config', str(config_path), '--datasets', task, '--device', 'cuda']
    result = subprocess.run(command)
    if result.returncode:
        # The parent sbatch redirects stdout/stderr to this log, retaining both attempts.
        text = log.read_text(errors='replace').lower()
        quota_error = any(s in text for s in ('disk quota exceeded', 'no space left on device',
                                              'errno 122', 'errno 28'))
        if quota_error and redirect(config_path):
            print('Retrying training once using the fallback checkpoint directory.', flush=True)
            result = subprocess.run(command)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
