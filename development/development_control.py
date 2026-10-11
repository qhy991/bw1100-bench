"""Owner-side observations shared by this finite development adapter."""
from pathlib import Path
import json
import math
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.json_stream import read_json_stream


def released(root):
    pending = [path for top in (root / 'campaign', root / '.local') for path in top.rglob('*.json')
               if not any(name in path.parts for name in ('triton-cache', 'aiter-jit-cache'))]
    seen = set()
    jobs = {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.is_symlink():
            return False
        try:
            records = ([json.loads(path.read_text())] if path.name.endswith('-terminal.json')
                       else read_json_stream(path))
        except (OSError, ValueError):
            return False
        for record in records:
            if not isinstance(record, dict):
                continue
            refs = record.get('admission_receipts', [])
            for ref in refs if isinstance(refs, list) else ():
                if not isinstance(ref, str):
                    continue
                referenced = root / ref
                if referenced.is_symlink():
                    return False
                try:
                    referenced = referenced.resolve()
                    referenced.relative_to(root.resolve())
                except (OSError, ValueError):
                    return False
                if not referenced.is_file():
                    return False
                pending.append(referenced)
            if record.get('schema') != 'bw1100-bench.hcu-admission.v1':
                continue
            job_id = record.get('job_id')
            if not isinstance(job_id, str) or not job_id:
                return False
            job = jobs.setdefault(job_id, {'admissions': [], 'terminals': []})
            required = ('completed_at', 'exit_code', 'container_still_running', 'after_vram', 'after_kfd_visible')
            if path.name.endswith('-terminal.json') or any(key in record for key in (
                    *required, 'status', 'reason', 'runtime_error_marker', 'release_observations')):
                after = record
            else:
                job['admissions'].append(record)
                terminal = path.with_name(path.stem + '-terminal.json')
                if terminal.is_symlink():
                    return False
                try:
                    after = json.loads(terminal.read_text())
                except (OSError, ValueError):
                    return False
            if (not isinstance(after, dict) or after.get('schema') != record['schema']
                    or after.get('job_id') != job_id or any(key not in after for key in required)
                    or after.get('container_still_running') is not False
                    or after.get('after_vram') != '0%' or after.get('after_kfd_visible') is not False):
                return False
            job['terminals'].append(after)
    for job in jobs.values():
        if any(any(key not in terminal or terminal[key] != value for key, value in admission.items())
               for admission in job['admissions'] for terminal in job['terminals']):
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
