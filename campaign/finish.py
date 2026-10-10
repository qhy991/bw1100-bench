"""Derive the attained budget endpoint and performance trajectory from receipts."""
import json
from pathlib import Path
import time
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.completion import confirmed_row
intake = json.loads((ROOT / 'campaign/intake.json').read_text())
plan = json.loads((ROOT / intake['plan']).read_text())
protocol = json.loads((ROOT / intake['protocol']).read_text())
deadline = json.loads((ROOT / 'campaign/deadline.json').read_text())
if time.time() < deadline['stop_at_epoch'] - 120:
    raise SystemExit('The fixed budget endpoint is not ready; continue useful search')
outcomes = []
unknown = []
for folder in sorted((ROOT / 'campaign/evaluations').iterdir()):
    if not folder.is_dir():
        continue
    try:
        outcomes.append(json.loads((folder / 'outcome.json').read_text()))
    except (ValueError, OSError) as error:
        unknown.append({'directory': str(folder.relative_to(ROOT)), 'reason': type(error).__name__})
accepted = []
stale = []
for d in outcomes:
    if d['status'] != 'accepted' or d['completed_at_epoch'] > deadline['search_stop_at_epoch']:
        continue
    bindings = dict(d.get('cake_artifact_bindings', {}))
    bindings[d['candidate_source']] = d['source_sha256']
    if any(not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
           for name, expected in bindings.items()):
        stale.append(d['id'])
        continue
    accepted.append(d)
ranking = lambda d: d['conservative_geomean']
best = max(accepted, key=ranking) if accepted else None
curve = []
for minute in protocol['checkpoint_minutes']:
    available = [d for d in accepted if d['completed_at_epoch'] <= deadline['started_at_epoch'] + minute*60]
    champion = max(available, key=ranking) if available else None
    curve.append({'minute': minute, 'best_qualified_id': champion['id'] if champion else None,
                  'speedup': champion['conservative_geomean'] if champion else 1.0,
                  'baseline_only': champion is None})
endpoint = {'arm': plan['arm'], 'task': plan['tasks'][0]['id'], 'protocol': intake['protocol'],
            'budget_hours': protocol['budget_hours'], 'evaluations': len(outcomes),
            'unknown_attempts': unknown, 'accepted_candidates': len(accepted), 'best_id': best['id'] if best else None,
            'stale_artifacts_excluded': stale,
            'best_speedup': best['conservative_geomean'] if best else 1.0, 'trajectory': curve,
            'measurement_scope': 'search result; final_confirmation separately decides within-budget confirmed performance'}
correct_times = [d['correctness_completed_at_epoch'] for d in outcomes if 'correctness_completed_at_epoch' in d]
endpoint['first_full_correct_seconds'] = min(correct_times)-deadline['started_at_epoch'] if correct_times else None
endpoint['first_robust_win_seconds'] = min(d['completed_at_epoch'] for d in accepted)-deadline['started_at_epoch'] if accepted else None
endpoint['experiment_kind'] = intake['experiment_kind']
endpoint['condition'] = intake['condition']
endpoint['replicate'] = intake['replicate']
endpoint['compiler_commit'] = intake['compiler_commit']
endpoint['frozen_workspace'] = intake['frozen_workspace']
cf=ROOT/'campaign/final-confirmation/result.json'
endpoint['final_confirmation']=json.loads(cf.read_text()) if cf.exists() else {'status':'missing'}
row = None
try:
    row = confirmed_row(ROOT, endpoint['final_confirmation'], accepted, plan)
    row['status'] = 'completed_pending_owner_review'
    row['reason'] = 'Fixed nominee passed the common assay; wrapper execution and rounding-chain attribution await independent owner review'
    endpoint['final_evidence_status'] = 'assay_qualified_pending_owner_review'
except (KeyError, ValueError, OSError) as error:
    endpoint['final_evidence_status'] = 'not_accepted'
    endpoint['final_evidence_reason'] = type(error).__name__ + ': ' + str(error)
if row is None:
    row = {'task': plan['tasks'][0]['id'],
           'status': 'blocked' if best or unknown or endpoint['final_confirmation'].get('status') == 'confirmed' else 'no_robust_gain',
           'reason': 'No confirmed candidate at the fixed endpoint; ' + endpoint['final_evidence_reason'],
           'evidence': ['campaign/ENDPOINT.json', 'campaign/JOURNAL.md'],
           'profile_skip_reason': 'Read JOURNAL for actual diagnostic attempts; no accepted candidate for final attribution'}
with (ROOT / 'campaign/ENDPOINT.json').open('x') as stream:
    json.dump(endpoint, stream, indent=2)
    stream.write('\n')
with (ROOT / 'campaign/DONE.json').open('x') as stream:
    json.dump({'tasks': [row]}, stream, indent=2)
    stream.write('\n')
print(json.dumps(endpoint, indent=2))
