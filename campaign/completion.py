"""Check durable completion evidence without changing experiment state."""
import hashlib
import json
from pathlib import Path
import subprocess


def inside(root, name):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def load(root, name):
    return json.loads(inside(root, name).read_text())


def validate(root, document, plan):
    rows = document['tasks']
    if len(rows) != len(plan['tasks']) or {r['task'] for r in rows} != {t['id'] for t in plan['tasks']}:
        raise ValueError('DONE must cover exactly the assigned tasks')
    definitions = {t['id']: t for t in plan['tasks']}
    for row in rows:
        if row['status'] not in ('accepted', 'no_robust_gain', 'blocked') or not row['reason'].strip():
            raise ValueError('each outcome needs its factual stopping reason')
        if not row['evidence']:
            raise ValueError('outcomes need durable evidence paths')
        for name in row['evidence']:
            if not inside(root, name).is_file():
                raise ValueError('missing evidence: ' + name)
        if not row.get('profile_evidence') and not row.get('profile_skip_reason', '').strip():
            raise ValueError('outcome needs diagnostic evidence or specific profiling skip reason')
        for name in row.get('profile_evidence', []):
            if not inside(root, name).is_file():
                raise ValueError('missing profile evidence')
        if row['status'] != 'accepted':
            continue
        gate = load(root, row['correctness_report'])
        if not (gate.get('full_device_correctness') and gate['status'] == 'passed'
                and gate['task'] == row['task'] and len(gate['cases']) == 160):
            raise ValueError('accepted candidate needs original full device gate')
        report = load(root, row['latency_report'])
        candidate = inside(root, row['candidate_source'])
        baseline = inside(root, definitions[row['task']]['baseline'])
        if hashlib.sha256(baseline.read_bytes()).hexdigest() != definitions[row['task']]['baseline_sha256']:
            raise ValueError('frozen baseline changed')
        if row['correctness_source_sha256'] != hashlib.sha256(candidate.read_bytes()).hexdigest():
            raise ValueError('correctness does not bind current candidate')
        if any(not c['passed'] for c in gate['cases']) or gate['arch'] != 'gfx938':
            raise ValueError('device gate contains failure or wrong architecture')
        if report['candidate_source_sha256'] != hashlib.sha256(candidate.read_bytes()).hexdigest():
            raise ValueError('timing does not bind current candidate')
        if report['baseline_source_sha256'] != hashlib.sha256(baseline.read_bytes()).hexdigest():
            raise ValueError('timing does not bind frozen baseline')
        if len(report['rows']) != 16:
            raise ValueError('timing must cover all original workloads')
        expected = {(c['workload_uuid'], c['round']) for c in gate['cases']}
        uuids = {c['workload_uuid'] for c in gate['cases']}
        if len(uuids) != 16 or expected != {(u, r) for u in uuids for r in range(10)}:
            raise ValueError('full gate must cover unique original UUID/round cells')
        if {c['workload_uuid'] for c in report['rows']} != uuids:
            raise ValueError('timing UUIDs differ from correctness workloads')
        wins = 0
        for cell in report['rows']:
            if not cell['inputs_unmutated_after_measurement']:
                raise ValueError('timed inputs mutated')
            base, cand = cell['baseline_us'], cell['candidate_us']
            bar = max(cell['aa_control_baseline_us']['abs_drift'] / base['median'], 0.01)
            ratios = [base[k] / cand[k] for k in ('forward_median', 'reverse_median')]
            if min(ratios) < 1 - bar:
                raise ValueError('material regression in a measurement direction')
            wins += min(ratios) - 1 > bar
        if not wins:
            raise ValueError('accepted candidate has no robust winning cell')


def completed(root):
    marker = root / 'campaign/DONE.json'
    if not marker.exists():
        return False
    try:
        intake = load(root, 'campaign/intake.json')
        plan = load(root, intake['plan'])
        validate(root, json.loads(marker.read_text()), plan)
        for folder in ('campaign/results', '.local/profile'):
            for path in (root / folder).rglob('*.json'):
                doc = json.loads(path.read_text())
                if not isinstance(doc, dict):
                    continue
                if doc.get('schema') != 'bw1100-bench.hcu-admission.v1' or 'completed_at' in doc:
                    continue
                terminal = path.with_name(path.stem + '-terminal.json')
                end = json.loads(terminal.read_text())
                if end['container_still_running'] or end['after_vram'] != '0%' or end['after_kfd_visible']:
                    raise ValueError('a GPU receipt has no released terminal state')
        ids = subprocess.check_output(['docker', 'ps', '-q'], universal_newlines=True, timeout=15).split()
        if ids:
            containers = json.loads(subprocess.check_output(['docker', 'inspect'] + ids,
                                     universal_newlines=True, timeout=15))
            if any(m['Source'] == str(root.resolve()) and m['Destination'] == '/work'
                   for c in containers for m in c.get('Mounts', [])):
                raise ValueError('an owned GPU/CPU container is still running')
        return True
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as error:
        print('DONE not accepted:', str(error), flush=True)
        return False
