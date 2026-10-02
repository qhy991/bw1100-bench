"""Derive the attained budget endpoint and performance trajectory from receipts."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
intake = json.loads((ROOT / 'campaign/intake.json').read_text())
plan = json.loads((ROOT / intake['plan']).read_text())
protocol = json.loads((ROOT / intake['protocol']).read_text())
deadline = json.loads((ROOT / 'campaign/deadline.json').read_text())
if time.time() < deadline['stop_at_epoch'] - 120:
    raise SystemExit('The fixed budget endpoint is not ready; continue useful search')
outcomes = [json.loads(p.read_text()) for p in (ROOT / 'campaign/evaluations').glob('*/outcome.json')]
accepted = [d for d in outcomes if d['status'] == 'accepted' and d['completed_at_epoch'] <= deadline['stop_at_epoch']]
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
            'best_speedup': best['conservative_geomean'] if best else 1.0, 'trajectory': curve,
            'measurement_scope': 'within-budget search; independent pair confirmation pending'}
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
