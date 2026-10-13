"""Verify an operator-owned classification of immutable diagnostic logs."""
import hashlib
import json
import re
from pathlib import Path

SCHEMA = 'bw1100-bench.hcu-admission.v1'

def reviewed_logs(root, review):
    if not review:
        return []
    if review.get('schema') != 'bw1100-bench.diagnostic-log-review.v1' or not review.get('authorization'):
        raise ValueError('Explicit diagnostic-log review required')
    exclusions = []
    for run in review['runs']:
        if Path(run['root']).resolve() != root.resolve():
            continue
        for log in run['logs']:
            def bound_file(relative, digest):
                path = root / relative
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Review file outside Run or symlink')
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError('Reviewed file changed')
                return raw
            raw = bound_file(log['path'], log['sha256']).decode()
            # These exact historical files are captured program output, not receipts.
            # Never exclude a file containing an admission schema, even malformed.
            if SCHEMA in raw:
                raise ValueError('Admission schema in diagnostic log')
            jobs = set()
            for pair in log['receipt_pairs']:
                admission = json.loads(bound_file(pair['admission'], pair['admission_sha256']))
                terminal = json.loads(bound_file(pair['terminal'], pair['terminal_sha256']))
                if admission.get('schema') != SCHEMA or not admission.get('job_id'):
                    raise ValueError('Invalid canonical admission')
                if any(terminal.get(k) != v for k, v in admission.items()):
                    raise ValueError('Terminal does not bind admission')
                if (terminal.get('container_still_running') is not False or terminal.get('after_vram') != '0%'
                        or terminal.get('after_kfd_visible') is not False or 'completed_at' not in terminal
                        or 'exit_code' not in terminal):
                    raise ValueError('Terminal does not confirm release')
                jobs.add(admission['job_id'])
            observed = set(re.findall(r'"job_id"\s*:\s*"([^"\r\n]+)"', raw))
            if not observed or not observed.issubset(jobs):
                raise ValueError('Diagnostic jobs lack matching canonical receipts')
            exclusions.append(log['path'])
    return exclusions
