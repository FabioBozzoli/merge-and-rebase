import errno
import importlib.util
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location('storage', Path(__file__).parents[1] / 'scripts/run_fewshot_with_storage_fallback.py')
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def test_quota_redirect_preserves_experiment_path(tmp_path, monkeypatch):
    primary, fallback = tmp_path / 'primary', tmp_path / 'fallback'
    primary.mkdir()
    monkeypatch.setattr(storage, 'PRIMARY', primary)
    monkeypatch.setattr(storage, 'FALLBACK', fallback)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'common': {'output': {'out_dir': str(primary / 'merge-and-rebase/K1/seed42')}}}))

    def quota(**kwargs):
        raise OSError(errno.EDQUOT, 'Disk quota exceeded')

    monkeypatch.setattr(storage.tempfile, 'TemporaryFile', quota)
    storage.preflight(config)
    assert yaml.safe_load(config.read_text())['common']['output']['out_dir'] == str(fallback / 'merge-and-rebase/K1/seed42')
    assert not storage.redirect(config)
