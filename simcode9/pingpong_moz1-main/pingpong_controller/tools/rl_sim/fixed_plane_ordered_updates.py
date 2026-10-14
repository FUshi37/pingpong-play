"""Authenticate explicitly requested ordered starts from completed CSV updates."""
import csv
import hashlib
from pathlib import Path


def verify_new_updates(run_directory: Path, *, stage_prefix: str, base_step: int,
                       minimum_new_updates: int, batch_steps: int = 131072) -> dict:
    """Exclude inherited history, duplicates and incompatible batch counts."""
    if minimum_new_updates<1 or batch_steps<1 or base_step<0:
        raise ValueError('Invalid ordered-update contract')
    path=run_directory/'curriculum_progress.csv'
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    steps=set()
    for row in rows:
        if not row.get('stage_name','').startswith(stage_prefix):
            continue
        try:
            step=int(row['global_step'])
            update=int(row['stage_update'])
        except (ValueError,TypeError,KeyError):
            continue
        if step>base_step and update>0:
            if (step-base_step)%batch_steps:
                raise ValueError('Incompatible completed update size')
            steps.add(step)
    ordered=sorted(steps)
    if len(ordered)<minimum_new_updates:
        raise ValueError(f'Need{minimum_new_updates} completed new updates; have{len(ordered)}')
    expected=[base_step+batch_steps*(i+1) for i in range(minimum_new_updates)]
    if ordered[:minimum_new_updates]!=expected:
        raise ValueError('Missing completed new updates in ordered history')
    return dict(run_directory=str(run_directory.resolve()),stage_prefix=stage_prefix,
        base_step=base_step,new_updates=len(ordered),required_new_updates=minimum_new_updates,
        fifteenth_or_required_step=ordered[minimum_new_updates-1],last_step=ordered[-1],
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
