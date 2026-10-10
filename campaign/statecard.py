"""Project this Run's own retained attempts for its fresh author context."""
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parents[1]


def render():
    rows = []
    for folder in sorted((ROOT / 'campaign/evaluations').iterdir()):
        if not folder.is_dir():
            continue
        path = folder / 'outcome.json'
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            result = {'status': 'unknown', 'reason': type(exc).__name__}
        rows.append({'directory': str(folder.relative_to(ROOT)), **result})
    value = {'deadline': json.loads((ROOT / 'campaign/deadline.json').read_text()),
             'observed_at_epoch': time.time(), 'own_evaluations': rows,
             'next_action': 'Use only this Run history; preserve unknown and failed attempts.'}
    output = ROOT / 'campaign/management/StateCard.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(output)
    return value
