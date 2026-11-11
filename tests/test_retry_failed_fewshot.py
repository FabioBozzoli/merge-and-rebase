import importlib.util
import json
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location('retry', Path(__file__).parents[1] / 'scripts/retry_failed_fewshot.py')
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def test_audit_uses_latest_attempt_and_matches_summary_to_config(tmp_path):
    folder = tmp_path / '.log/vitb16_full_fewshot/GTSRB/1_shots'
    folder.mkdir(parents=True)
    for job, status in [(100, 'failed'), (101, 'success')]:
        cfg = folder / f'seed_42_job_{job}_0_config.yaml'
        cfg.write_text('common: {}')
        run = {'run_logging': {'metadata': {'config_path': str(cfg)}, 'status': status, 'created_unix': job}}
        (folder / f'run_{job}.json').write_text(json.dumps(run))
    records = retry.audit(tmp_path)
    assert len(records) == 1
    assert records[0]['status'] == 'success'
    assert records[0]['previous_job'] == 101
    (folder / 'seed_42_job_102_0_config.yaml').write_text('common: {}')
    assert retry.audit(tmp_path)[0]['status'] == 'missing_summary'


def test_retry_preserves_exact_saved_config_and_resource_headers(tmp_path):
    folder = tmp_path / '.log/family/GTSRB/1_shots'
    folder.mkdir(parents=True)
    source = tmp_path / 'scripts'
    source.mkdir()
    (source / 'original.sbatch').write_text('#!/bin/bash\n#SBATCH --mem=32G\n#SBATCH --array=0-39\n')
    cfg = folder / 'seed_42_job_100_15_config.yaml'
    cfg.write_text('common:\n  seed: 42\n  train:\n    lr: 0.00001\n'
                   '  output:\n    out_dir: /work/intesasanpaolo_phd/merge-and-rebase/checkpoints/full_fewshot/K1/seed42\n'
                   'datasets_order: [GTSRB]\n')
    record = dict(family='family', script='original.sbatch', task='GTSRB', shots='1_shots', seed=42,
                  previous_job=100, index=15, config=str(cfg))
    old_root = retry.ROOT
    retry.ROOT = tmp_path
    try:
        script = retry.retry_script(record, tmp_path)
    finally:
        retry.ROOT = old_root
    text = script.read_text()
    assert '#SBATCH --mem=32G' in text
    assert '#SBATCH --array=' not in text
    assert 'export HF_HOME=' not in text
    assert 'common:' in script.with_suffix('.yaml').read_text()
    assert 'run_fewshot_with_storage_fallback.py' in text
    assert ' GTSRB ' in text
    saved = yaml.safe_load(script.with_suffix('.yaml').read_text())
    assert saved['common']['output']['out_dir'] == '/work/tesi_bcalderara/merge-and-rebase/checkpoints/full_fewshot/K1/seed42'
    assert saved['common']['train']['lr'] == 0.00001
    assert yaml.safe_load(cfg.read_text())['common']['output']['out_dir'].startswith('/work/intesasanpaolo_phd/')


@pytest.mark.parametrize('root', ['/work/bcalde_phd', '/work/tesi_bcalderara'])
def test_retry_redirects_fallback_and_already_redirected_paths(tmp_path, monkeypatch, root):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/original.sbatch').write_text('#!/bin/bash\n#SBATCH --mem=32G\n')
    config = tmp_path / 'original.yaml'
    config.write_text(yaml.safe_dump({'datasets_order': ['DTD'], 'common': {
        'seed': 42, 'output': {'out_dir': root + '/merge-and-rebase/checkpoints/lora/K10/seed42'}}}))
    monkeypatch.setattr(retry, 'ROOT', tmp_path)
    record = dict(family='family', script='original.sbatch', task='DTD', shots='10_shots', seed=42,
                  previous_job=100, index=1, config=str(config))
    script = retry.retry_script(record, tmp_path)
    output = yaml.safe_load(script.with_suffix('.yaml').read_text())['common']['output']['out_dir']
    assert output == '/work/tesi_bcalderara/merge-and-rebase/checkpoints/lora/K10/seed42'
