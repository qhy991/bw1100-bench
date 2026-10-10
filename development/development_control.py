"""Owner-side observations shared by this finite development adapter."""
from pathlib import Path
import json
import math
import re


def released(root):
    for top in (root / 'campaign', root / '.local'):
        for path in top.rglob('*.json'):
            if path.name.endswith('-terminal.json') or any(
                    name in path.parts for name in ('triton-cache', 'aiter-jit-cache')):
                continue
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                return False
            if not isinstance(record, dict) or record.get('schema') != 'bw1100-bench.hcu-admission.v1' or 'completed_at' in record:
                continue
            terminal = path.with_name(path.stem + '-terminal.json')
            try:
                after = json.loads(terminal.read_text())
            except (OSError, ValueError):
                return False
            if (after.get('container_still_running') is not False
                    or after.get('after_vram') != '0%' or after.get('after_kfd_visible') is not False):
                return False
    return True


def nominee(root, compiler):
    accepted = []
    for path in sorted((root / 'campaign/evaluations').glob('*/outcome.json')):
        outcome = json.loads(path.read_text())
        samples = outcome.get('primary_us', [])
        if (outcome.get('status') == 'accepted' and outcome.get('phase') == 'search'
                and outcome.get('compiler') == compiler and samples
                and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in samples)):
            accepted.append(outcome)
    return min(accepted, key=lambda x: sorted(x['primary_us'])[len(x['primary_us']) // 2]) if accepted else None


def close_request(root, best):
    path = root / 'campaign/SEARCH_CLOSE.json'
    if not path.exists():
        return None
    request = json.loads(path.read_text())
    if (not isinstance(request, dict) or set(request) != {'candidate_id', 'rationale'}
            or not isinstance(request['candidate_id'], str)
            or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', request['candidate_id'])
            or not isinstance(request['rationale'], str) or not request['rationale'].strip()
            or best is None or request['candidate_id'] != best['id']):
        raise ValueError('search-close request needs the current accepted nominee and an evidence-bound rationale')
    return request
