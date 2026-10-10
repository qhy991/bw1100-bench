"""Own one fresh Bench author session and its original bounded final confirmation."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.binding import reconcile


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def author(command, env, timeout, log):
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        return 124


def main():
    os.chdir(ROOT)
    binding = reconcile(ROOT)
    if sys.argv[1:] == ['--check']:
        print('Prepared binding matches the frozen allocation')
        return
    if sys.argv[1:]:
        raise ValueError('launch takes only optional --check')
    now = time.time()
    controls = binding['author']
    wall, reserve = controls['wall_time_seconds'], controls['confirmation_seconds']
    deadline = {'started_at_epoch': now, 'search_stop_at_epoch': now + wall - reserve,
                'stop_at_epoch': now + wall, 'wall_time_seconds': wall,
                'confirmation_seconds': reserve, 'selected_hcu': binding['hcu'],
                'preparation_cost_included': True}
    write_new(ROOT / 'campaign/deadline.json', deadline)
    intake = {**binding, 'plan': 'campaign/groups/c.json', 'protocol': 'campaign/protocol.json',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'experiment_kind': 'rolling_bench_fresh_search', 'arm': 'cake_ir',
              'model': controls['model'], 'budget_hours': wall / 3600}
    write_new(ROOT / 'campaign/intake.json', intake)
    env = dict(os.environ, HOME=binding['home'], HIP_VISIBLE_DEVICES=str(binding['hcu']))
    env['PATH'] = str(Path(binding['home']) / '.local/bin') + ':' + env.get('PATH', '')
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        env.pop(key, None)
    model_code, error = None, None
    try:
        profile = subprocess.check_output([sys.executable, 'scripts/ralph_profile_intake.py',
                                          'campaign/profile-intake.json'], env=env, text=True)
        for command in (
            ['python3', 'bwbench.py', 'audit', '--output', 'campaign/results/intake-audit.json'],
            ['python3', 'campaign/compiler_preflight.py'],
        ):
            remaining = int(deadline['search_stop_at_epoch'] - time.time())
            if remaining <= 10:
                raise ValueError('No preparation time remains before search cutoff')
            subprocess.run(['bash', 'scripts/dtk.sh', 'cpu', binding['image'],
                            '/usr/bin/timeout', '-k', '10s', str(min(180, remaining - 10)) + 's',
                            *command], env=env, check=True)
        remaining = int(deadline['search_stop_at_epoch'] - time.time())
        if remaining <= 0:
            raise ValueError('Preparation consumed the search budget')
        with (ROOT / 'campaign/logs/ralph.log').open('x') as log:
            model_code = author(['hmz', 'exec', '-f', 'campaign/ralph_flow.py',
                                 '-a', controls['model'], '-c', 'campaign/budget.yaml',
                                 profile + '\n' + (ROOT / 'campaign/TASK.md').read_text()],
                                env, remaining, log)
        with (ROOT / 'campaign/logs/final-confirmation-owner.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'campaign.final_confirmation'], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=max(1, deadline['stop_at_epoch'] - time.time()))
        time.sleep(max(0, deadline['stop_at_epoch'] - 120 - time.time()))
        subprocess.run([sys.executable, 'campaign/finish.py'], env=env, check=True)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        write_new(ROOT / 'campaign/controller-status.json', {
            'model_exit_code': model_code, 'error': error,
            'done_exists': (ROOT / 'campaign/DONE.json').exists(),
            'owner_finalized_at_epoch': time.time()})


if __name__ == '__main__':
    main()
