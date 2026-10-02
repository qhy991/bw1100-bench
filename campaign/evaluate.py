"""Freeze and evaluate one candidate through the common correctness/wall path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.completion import validate, inside


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--cake-artifact', action='append', default=[])
    parser.add_argument('--profile-evidence', action='append', default=[])
    parser.add_argument('--profile-skip-reason', default='')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', args.id):
        parser.error('id must be a unique lowercase artifact name')
    os.chdir(str(ROOT))
    intake = json.loads((ROOT / 'campaign/intake.json').read_text())
    plan = json.loads((ROOT / intake['plan']).read_text())
    deadline = json.loads((ROOT / 'campaign/deadline.json').read_text())
    folder = ROOT / 'campaign/evaluations' / args.id
    folder.mkdir(parents=True, exist_ok=False)
    frozen = ROOT / 'campaign/candidates' / ('frozen_' + args.id + '.py')
    with frozen.open('x') as stream:
        stream.write(inside(ROOT, args.candidate).read_text())
    source_hash = digest(frozen)
    receipts = []
    if plan['arm'] == 'cake_ir':
        if not args.cake_artifact:
            raise ValueError('Cake candidates need their executed emission receipts')
        command = ['bash', 'scripts/dtk.sh', 'cpu', intake['image'],
                   '/usr/bin/timeout', '-k', '10s', '180s', 'python3',
                   'campaign/cake_bridge.py', 'verify-many'] + args.cake_artifact
        subprocess.check_call(command)
    task = plan['tasks'][0]['id']
    relative = lambda p: str(p.relative_to(ROOT))
    gate = folder / 'correctness.json'
    latency = folder / 'latency.json'
    env = dict(os.environ, HOME='/data3/testuser01', HIP_VISIBLE_DEVICES=str(plan['hcu']), BWBENCH_TIMEOUT='900')
    explicit_env = ['env', 'AITER_JIT_DIR=/work/.local/aiter-jit-cache',
                    'TRITON_CACHE_DIR=/work/.local/triton-cache']
    source_in_device = '/work/' + relative(frozen)
    stages = [('correctness', ['python3', 'bwbench.py', 'check', '--task', task,
                 '--device', 'cuda:0', '--candidate', source_in_device,
                 '--workloads', 'all', '--rounds', '10', '--output', relative(gate)]),
              ('latency', ['python3', 'campaign/paired_wall.py', '--candidate', source_in_device,
                 '--gate', relative(gate), '--output', relative(latency)])]
    outcome = {'id': args.id, 'task': task, 'arm': plan['arm'], 'started_at_epoch': time.time(),
               'candidate_source': relative(frozen), 'source_sha256': source_hash,
               'status': 'not_qualified', 'cake_artifacts': args.cake_artifact}
    try:
        for stage, command in stages:
            receipt = folder / (stage + '-admission.json')
            receipts.append(relative(receipt))
            result = subprocess.call(['bash', 'campaign/admit.sh', 'gpu', intake['image'],
                                      relative(receipt)] + explicit_env + command, env=env)
            if result:
                raise ValueError(stage + ' gate failed or admission unavailable: exit ' + str(result))
            terminal = receipt.with_name(receipt.stem + '-terminal.json')
            outcome[stage + '_completed_at_epoch'] = json.loads(terminal.read_text())['completed_at']
        if digest(frozen) != source_hash:
            raise ValueError('frozen candidate changed during evaluation')
        for name in receipts:
            terminal = inside(ROOT, name).with_name(Path(name).stem + '-terminal.json')
            record = json.loads(terminal.read_text())
            if record['completed_at'] > deadline['stop_at_epoch']:
                raise ValueError('evaluation finished after the search budget')
        row = {'task': task, 'status': 'accepted', 'reason': 'common full gate and paired wall timing',
               'evidence': [relative(gate), relative(latency)],
               'candidate_source': relative(frozen), 'correctness_source_sha256': source_hash,
               'correctness_report': relative(gate), 'latency_report': relative(latency),
               'profile_evidence': args.profile_evidence,
               'profile_skip_reason': args.profile_skip_reason,
               'cake_artifacts': args.cake_artifact}
        validate(ROOT, {'tasks': [row]}, plan)
        timing = json.loads(latency.read_text())
        ratios = [min(c['baseline_us'][k] / c['candidate_us'][k]
                      for k in ('forward_median', 'reverse_median')) for c in timing['rows']]
        outcome.update(status='accepted', conservative_geomean=math.exp(sum(map(math.log, ratios))/len(ratios)),
                       speedup_min=min(ratios), speedup_max=max(ratios), handoff_row=row)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        outcome['reason'] = str(error)
    outcome.update(completed_at_epoch=time.time(), admission_receipts=receipts)
    with (folder / 'outcome.json').open('x') as stream:
        json.dump(outcome, stream, indent=2)
        stream.write('\n')
    print(json.dumps(outcome, indent=2))
    return 0 if outcome['status'] == 'accepted' else 1


if __name__ == '__main__':
    sys.exit(main())
