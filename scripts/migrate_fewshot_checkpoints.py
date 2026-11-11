"""Move quiescent checkpoint directories, validating content before deleting originals."""
import shutil
import subprocess
from pathlib import Path

SOURCE = Path('/work/intesasanpaolo_phd/merge-and-rebase/checkpoints')
DESTINATION = Path('/work/bcalde_phd')
NAMES = ('full_fewshot', 'full_fewshot_vitl14_datacomp', 'lora_fewshot_vitl14_datacomp')


def main():
    queue = subprocess.run(['squeue', '-h', '-o', '%u|%j'], check=True, text=True, capture_output=True)
    writers = {'vitb16_full_fewshot', 'vitb16_laion2b_lora_fewshot',
               'vitl14_datacomp_full_fewshot', 'vitl14_datacomp_lora_fewshot'}
    if any(line.split('|')[0] == 'bcalderara' and line.split('|')[-1] in writers
           for line in queue.stdout.splitlines()):
        raise RuntimeError('Fine-tuning writers still active; refusing migration')
    for name in NAMES:
        source, dest = SOURCE / name, DESTINATION / name
        if source.is_symlink():
            if source.resolve() != dest.resolve():
                raise RuntimeError(f'Unexpected existing symlink: {source}')
            print(f'Already migrated: {source} -> {dest}', flush=True)
            continue
        if not source.is_dir():
            raise RuntimeError(f'Missing source: {source}')
        # An existing unrelated destination must never be merged or overwritten.
        marker = DESTINATION / f'.{name}.migration_in_progress'
        if dest.exists() and not marker.exists():
            raise RuntimeError(f'Destination already exists: {dest}')
        marker.write_text(str(source))
        dest.mkdir(exist_ok=True)
        print(f'Copying {source} -> {dest}', flush=True)
        subprocess.run(['rsync', '-rltp', '--checksum', str(source) + '/', str(dest) + '/'], check=True)
        check = subprocess.run(['rsync', '-rltpcn', '--delete', '--itemize-changes',
                                str(source) + '/', str(dest) + '/'],
                               check=True, text=True, capture_output=True)
        if check.stdout.strip():
            raise RuntimeError(f'Copy verification failed: {check.stdout}')
        # With all writers finished and retries dependent on this job, replace the
        # original directory with a link before freeing the verified original data.
        backup = source.with_name(source.name + '.migration_original')
        if backup.exists():
            raise RuntimeError(f'Backup already exists: {backup}')
        source.rename(backup)
        try:
            source.symlink_to(dest, target_is_directory=True)
        except BaseException:
            backup.rename(source)
            raise
        shutil.rmtree(backup)
        marker.unlink()
        print(f'Migrated and verified: {source} -> {dest}', flush=True)


if __name__ == '__main__':
    main()
