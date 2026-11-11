import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('migration', Path(__file__).parents[1] / 'scripts/migrate_fewshot_checkpoints.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def setup(tmp_path, monkeypatch, queue=''):
    source, dest = tmp_path / 'source', tmp_path / 'dest'
    source.mkdir()
    dest.mkdir()
    for name in migration.NAMES:
        (source / name / 'K1').mkdir(parents=True)
        (source / name / 'K1/model.pt').write_bytes(b'checkpoint\x00content')
    monkeypatch.setattr(migration, 'SOURCE', source)
    monkeypatch.setattr(migration, 'DESTINATION', dest)
    original = subprocess.run

    def run(command, **kwargs):
        if command[0] == 'squeue':
            return subprocess.CompletedProcess(command, 0, stdout=queue)
        return original(command, **kwargs)

    monkeypatch.setattr(migration.subprocess, 'run', run)
    return source, dest


def test_migration_verifies_and_keeps_old_paths_working(tmp_path, monkeypatch):
    source, dest = setup(tmp_path, monkeypatch)
    migration.main()
    for name in migration.NAMES:
        assert (source / name).is_symlink()
        assert (dest / name / 'K1/model.pt').read_bytes() == b'checkpoint\x00content'
        assert (source / name / 'K1/model.pt').read_bytes() == b'checkpoint\x00content'
        assert not (source / (name + '.migration_original')).exists()
    migration.main()


def test_active_writer_blocks_migration(tmp_path, monkeypatch):
    source, dest = setup(tmp_path, monkeypatch, 'bcalderara|vitl14_datacomp_full_fewshot\n')
    with pytest.raises(RuntimeError, match='still active'):
        migration.main()
    assert not list(dest.iterdir())
    assert all(not (source / name).is_symlink() for name in migration.NAMES)
