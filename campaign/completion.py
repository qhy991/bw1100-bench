"""Check durable completion evidence without changing experiment state."""
import hashlib
import json
from pathlib import Path


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
        caller = load(root, row['caller_report'])
        precision = load(root, row['precision_report'])
        expected_checks = {'fresh_call', 'same_tensor_object_new_storage', 'retained_output_unchanged', 'fresh_view_same_storage'}
        if caller['task'] != row['task'] or caller['workload_uuid'] != definitions[row['task']]['smoke_uuid'] or len(caller['checks']) != 4 or {c['name'] for c in caller['checks']} != expected_checks:
            raise ValueError('caller checks differ from frozen Task/case contract')
        if caller['status'] != 'passed' or not all(c['passed'] for c in caller['checks']) or precision['status'] != 'passed':
            raise ValueError('caller or precision qualification failed')
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
        if caller['candidate_source_sha256'] != hashlib.sha256(candidate.read_bytes()).hexdigest():
            raise ValueError('caller guards do not bind current candidate')
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


def confirmed_row(root, confirmation, accepted, plan):
    """Return only the unchanged nominee covered by the final confirmation."""
    if confirmation.get('status') != 'confirmed':
        raise ValueError('Final confirmation did not pass')
    nominees = [value for value in accepted if value['id'] == confirmation.get('nominated_id')]
    if len(nominees) != 1:
        raise ValueError('Confirmed nomination is missing, stale or ambiguous')
    nominee = load(root, 'campaign/final-confirmation/nomination.json')
    if nominee != nominees[0]:
        raise ValueError('Final nomination differs from its retained search outcome')
    bindings = dict(nominee.get('cake_artifact_bindings', {}))
    bindings[nominee['candidate_source']] = nominee['source_sha256']
    if any(hashlib.sha256(inside(root, name).read_bytes()).hexdigest() != expected
           for name, expected in bindings.items()):
        raise ValueError('Confirmed nomination source changed')
    row = dict(nominee['handoff_row'], latency_report='campaign/final-confirmation/latency.json')
    validate(root, {'tasks': [row]}, plan)
    row['reason'] = 'Source-bound fixed nominee passed final confirmation; ' + row['reason']
    return row
