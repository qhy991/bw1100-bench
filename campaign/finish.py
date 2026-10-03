"""Derive the attained budget endpoint and performance trajectory from receipts."""
import json
from pathlib import Path
import time
import hashlib

ROOT = Path(__file__).resolve().parents[1]
intake = json.loads((ROOT / 'campaign/intake.json').read_text())
plan = json.loads((ROOT / intake['plan']).read_text())
protocol = json.loads((ROOT / intake['protocol']).read_text())
deadline = json.loads((ROOT / 'campaign/deadline.json').read_text())
if time.time() < deadline['stop_at_epoch'] - 120:
    raise SystemExit('The fixed budget endpoint is not ready; continue useful search')
outcomes = [json.loads(p.read_text()) for p in (ROOT / 'campaign/evaluations').glob('*/outcome.json')]
accepted = []
stale = []
for d in outcomes:
    if d['status'] != 'accepted' or d['completed_at_epoch'] > deadline['stop_at_epoch']:
        continue
    bindings = dict(d.get('cake_artifact_bindings', {}))
    bindings[d['candidate_source']] = d['source_sha256']
    if any(not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
           for name, expected in bindings.items()):
        stale.append(d['id'])
        continue
    accepted.append(d)
best = max(accepted, key=lambda d: d['conservative_geomean']) if accepted else None
curve = []
for minute in protocol['checkpoint_minutes']:
    available = [d for d in accepted if d['completed_at_epoch'] <= deadline['started_at_epoch'] + minute*60]
    champion = max(available, key=lambda d: d['conservative_geomean']) if available else None
    curve.append({'minute': minute, 'best_qualified_id': champion['id'] if champion else None,
                  'speedup': champion['conservative_geomean'] if champion else 1.0,
                  'baseline_only': champion is None})
endpoint = {'arm': plan['arm'], 'task': plan['tasks'][0]['id'], 'protocol': intake['protocol'],
            'budget_hours': protocol['budget_hours'], 'evaluations': len(outcomes),
            'accepted_candidates': len(accepted), 'best_id': best['id'] if best else None,
            'stale_artifacts_excluded': stale,
            'best_speedup': best['conservative_geomean'] if best else 1.0, 'trajectory': curve,
            'measurement_scope': 'within-budget search; independent pair confirmation pending'}
correct_times = [d['correctness_completed_at_epoch'] for d in outcomes if 'correctness_completed_at_epoch' in d]
endpoint['first_full_correct_seconds'] = min(correct_times)-deadline['started_at_epoch'] if correct_times else None
endpoint['first_robust_win_seconds'] = min(d['completed_at_epoch'] for d in accepted)-deadline['started_at_epoch'] if accepted else None
seed = next((d for d in accepted if d['id'] == 'seed00'), None)
if intake.get('round') == 2:
    endpoint['round'] = 2
    endpoint['parent'] = intake['parent']
    endpoint['seed00_speedup'] = seed['conservative_geomean'] if seed else None
    endpoint['best_over_seed_score_ratio'] = best['conservative_geomean']/seed['conservative_geomean'] if best and seed else None
    endpoint['incremental_gain_requires_confirmation'] = True
with (ROOT / 'campaign/ENDPOINT.json').open('x') as stream:
    json.dump(endpoint, stream, indent=2)
    stream.write('\n')
if best:
    row = best['handoff_row']
    row['reason'] = 'Highest conservative geomean among canonical accepted evaluations within fixed budget; ' + row['reason']
else:
    row = {'task': plan['tasks'][0]['id'], 'status': 'no_robust_gain',
           'reason': 'No canonical accepted candidate before fixed budget endpoint; baseline remains the incumbent',
           'evidence': ['campaign/ENDPOINT.json', 'campaign/JOURNAL.md'],
           'profile_skip_reason': 'Read JOURNAL for actual diagnostic attempts; no accepted candidate for final attribution'}
with (ROOT / 'campaign/DONE.json').open('x') as stream:
    json.dump({'tasks': [row]}, stream, indent=2)
    stream.write('\n')
print(json.dumps(endpoint, indent=2))
