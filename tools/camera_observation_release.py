#!/usr/bin/env python3
"""Verify or restore a frozen camera observation release, never start hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def verify(release):
    release=Path(release).resolve()
    manifest=json.loads((release/'manifest.json').read_text())
    for item in manifest['files']:
        path=(release/item['stored_path']).resolve()
        if not path.is_relative_to(release):
            raise ValueError('Release path escapes its directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Release checksum mismatch: '+item['stored_path'])
    return manifest


def restore(release,workspace):
    release=Path(release).resolve();workspace=Path(workspace).resolve()
    manifest=verify(release)
    selected=[item for item in manifest['files'] if item.get('restore')]
    destinations=[]
    for item in selected:
        target=(workspace/item['workspace_path']).resolve()
        if not target.is_relative_to(workspace):
            raise ValueError('Restore path escapes the workspace')
        destinations.append((item,target))
    # Back up every destination before overwriting any file. No git resets,
    # control scheduling, raw recordings or unrelated files are involved.
    backup=workspace/'analysis_results/camera_release_restore'/str(time.time_ns())
    for item,target in destinations:
        if target.exists():
            saved=backup/item['workspace_path'];saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,saved)
    for item,target in destinations:
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.release-tmp')
        shutil.copy2(release/item['stored_path'],temporary)
        temporary.replace(target)
    print(json.dumps(dict(version=manifest['version'],restored=len(selected),backup=str(backup)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('release',type=Path)
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--restore',action='store_true',help='Restore only listed ball-observation files; stop the probe first.')
    args=p.parse_args()
    if args.restore:restore(args.release,args.workspace)
    else:
        manifest=verify(args.release)
        print(json.dumps(dict(version=manifest['version'],verified_files=len(manifest['files'])),ensure_ascii=False))
