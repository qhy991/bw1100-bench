"""Create one immutable group intake and its wall-clock deadline."""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('group', choices=list('abcdef'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
plan_path = 'campaign/groups/' + args.group + '.json'
plan = json.loads((root / plan_path).read_text())
now = int(time.time())
for path, doc in (
    ('campaign/intake.json', {'plan': plan_path, 'hcu': plan['hcu'],
      'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(root), universal_newlines=True).strip(),
      'image': 'sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260',
      'arm': plan['arm'], 'compiler_commit': plan['compiler_commit'],
      'model': 'claude/glm-5.3:high', 'flow': 'campaign/ralph_flow.py',
      'official_flow_sha256': 'dff76869f6c823cb6136e0647d4a52777d1580bb05eff36806211182c946a852'}),
    ('campaign/deadline.json', {'started_at_epoch': now, 'stop_at_epoch': now + 10800,
                               'budget_hours': 3, 'selected_hcu': plan['hcu']}),
):
    with (root / path).open('x') as stream:
        json.dump(doc, stream, indent=2)
        stream.write('\n')

