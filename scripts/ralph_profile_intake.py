#!/usr/bin/env python3
"""Bind a new Ralph campaign to a readable DCU profiling skill."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_intake(root, home, output):
    root = root.resolve()
    output = root / output
    try:
        output.resolve().relative_to(root)
    except ValueError:
        raise ValueError('intake path must stay inside the repository')
    skill = home / '.agents/skills/dcu-rocprof-report-skill/SKILL.md'
    guide = root / 'docs/RALPH-PROFILING.md'
    if not skill.is_file() or not os.access(skill, os.R_OK):
        raise FileNotFoundError('DCU profiling skill is not readable: ' + str(skill))
    if not guide.is_file():
        raise FileNotFoundError('Ralph profiling guide is missing: ' + str(guide))
    record = {
        'schema': 'bw1100-bench.ralph-profile-intake.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'skill_path': str(skill.resolve()),
        'skill_sha256': sha256(skill),
        'guide_path': str(guide),
        'guide_sha256': sha256(guide),
        'agent_read_status': 'required_unverified',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return (
        'First-round required action: use the Read tool to open '
        + record['skill_path'] + ' (SHA-256 ' + record['skill_sha256']
        + ') and ' + record['guide_path']
        + ' before changing any kernel. Record the Read in the campaign journal. '
        'A readable path or this intake receipt alone does not prove the skill was read. '
        'For a qualified community baseline, collect a bounded target-kernel rocprof CSV '
        'through scripts/rocprof.sh, or record a specific profile_unavailable or '
        'not_decision_relevant reason. Never use profiled duration as a speed score.\n\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('intake', type=Path, help='create-only path inside this repository')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(create_intake(root, Path.home(), args.intake), end='')


if __name__ == '__main__':
    main()
