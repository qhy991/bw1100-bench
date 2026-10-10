"""One fixed three-hour development Run; root coordinator owns author slots."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from development_control import close_request, nominee, released
from development_stop import stopped, halt
from development_binding import reconcile

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from campaign.launch import author_entry


def validate_nominee(root, binding, best, environment, stop):
    remaining = int(stop - time.time()) - 20
    if remaining <= 0:
        raise TimeoutError('no budget remains to validate the fixed nominee')
    result = subprocess.run(['bash', 'scripts/dtk.sh', 'cpu', binding['image'], 'python3',
        'campaign/development_evaluate.py', '--candidate', best['candidate'], '--id', best['id'],
        '--phase', 'validate'], cwd=root, env=environment, capture_output=True, text=True,
        timeout=min(180, remaining))
    if result.returncode:
        raise ValueError('source-bound nominee validation failed: ' + result.stderr[-1000:])


def run_author(root, binding, deadline, environment, instruction, entry):
    remaining = int(deadline['search_stop_at_epoch'] - time.time())
    if remaining <= 0:
        raise TimeoutError('CPU preparation consumed the search interval')
    command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(remaining) + 's',
        str(entry), 'exec', '-f', 'campaign/ralph_flow.py', '-a', binding['model'],
        '-c', 'campaign/budget-3h.yaml', instruction + '\n' + (root / 'campaign/TASK.md').read_text()]
    checked = None
    approved = None
    with (root / 'campaign/logs/ralph.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                if stopped(binding):
                    # Active tools finish; Ralph returns at the next model-call boundary.
                    break
                request_path = root / 'campaign/SEARCH_CLOSE.json'
                if request_path.exists() and time.time() < deadline['search_stop_at_epoch'] and released(root):
                    text = request_path.read_text()
                    ready_path = root / 'campaign/author-round-terminal.json'
                    try:
                        ready = json.loads(ready_path.read_text())
                    except (OSError, ValueError):
                        ready = {}
                    if ready.get('request') != text:
                        time.sleep(1)
                        continue
                    if text != checked:
                        checked = text
                        decision = dict(node_epoch=time.time(), request=text)
                        try:
                            best = nominee(root, binding['compiler'])
                            request = close_request(root, best)
                            validate_nominee(root, binding, best, environment, deadline['search_stop_at_epoch'])
                            if not released(root):
                                checked = None
                                continue
                            approved = dict(request, nominee=best['id'], node_epoch=time.time(),
                                            compiler=binding['compiler'], admission_release_verified=True)
                            decision.update(decision='approved', validated=approved)
                        except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired) as error:
                            decision.update(decision='refused', reason=str(error))
                        with (root / 'campaign/search-close-owner.jsonl').open('a') as record:
                            record.write(json.dumps(decision) + '\n')
                        current = root / 'campaign/search-close-owner.json'
                        temporary = current.with_suffix('.json.tmp')
                        temporary.write_text(json.dumps(decision, indent=2))
                        temporary.replace(current)
                        if approved:
                            # The agent round already returned; ralph_flow now returns naturally.
                            # No model tool or GPU client is interrupted for early closure.
                            break
                time.sleep(1)
        finally:
            # Keep the author slot owned even if this observer fails. The existing
            # search deadline bounds the child; never interrupt an active GPU client.
            code = process.wait()
        return code, approved


def write_terminal(root, binding, model_exit, best, confirmation, failure, closure):
    release = released(root)
    endpoint = dict(scope='Compiler-development Task; diagnostic samples are not a performance win',
        task=binding['task'], compiler=binding['compiler'], gateway=binding['gateway'],
        wall_time_seconds=10800, model_exit=model_exit, best=best, confirmation=confirmation,
        error=failure, search_close=closure, token_limits=None, device_release_verified=release,
        evidence_owner='canonical Task workload/oracle/comparator; own emissions and gateway receipts',
        ended_node_epoch=time.time())
    status = ('failed' if failure else 'completed' if best and confirmation
              and confirmation.get('status') == 'accepted' else 'no_qualified_result')
    records = {'ENDPOINT.json': endpoint,
               'DONE.json': dict(terminal=True, status=status, scope=endpoint['scope'],
                                 device_release_verified=release),
               'controller-status.json': dict(exit_code=1 if failure else 0, done_exists=True,
                   device_release_verified=release, ended_node_epoch=time.time())}
    for name, value in records.items():
        with (root / 'campaign' / name).open('x') as stream:
            json.dump(value, stream, indent=2)


def main(root=R, check=False):
    os.chdir(root)
    binding = reconcile(root)
    entry, environment = author_entry(binding)
    if check:
        print('Prepared development contract and absolute author entry are available')
        return 0
    if stopped(binding):
        raise RuntimeError('Assigned HCU is halted; no new Run intake')
    started = time.time()
    deadline = dict(started_at_epoch=started, search_stop_at_epoch=started + 9000,
                    stop_at_epoch=started + 10800, wall_time_seconds=10800, budget_hours=3,
                    confirmation_seconds=1800, author_executable=str(entry))
    # Duplicate launches must not modify an earlier Run or its terminal records.
    with (root / 'campaign/deadline.json').open('x') as stream:
        json.dump(deadline, stream, indent=2)
    environment['BWBENCH_STOP_FILE'] = binding['batch_stop_file']
    phase = 'profiling_intake'
    model_exit = best = confirmation = failure = closure = None
    try:
        if stopped(binding):
            raise RuntimeError('batch halted before preparation')
        instruction = subprocess.check_output(['python3', 'scripts/ralph_profile_intake.py',
            'campaign/profile-intake.json'], cwd=root, env=environment, text=True, timeout=180)
        phase = 'cpu_preparation'
        seconds = min(1800, max(1, int(deadline['search_stop_at_epoch'] - time.time())))
        with (root / 'campaign/logs/preparation.log').open('x') as log:
            subprocess.run(['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(seconds) + 's',
                'bash', 'scripts/dtk.sh', 'cpu', binding['image'], 'python3', 'campaign/development_prepare.py'],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
        phase = 'authoring'
        model_exit, closure = run_author(root, binding, deadline, environment, instruction, entry)
        if stopped(binding):
            raise RuntimeError('batch halted; no final device confirmation admitted')
        if not released(root):
            halt(binding, 'author ended with unknown or unreleased admission', str(root))
            raise RuntimeError('author ended with an unreleased or unknown device admission; hold this slot')
        if model_exit not in (0, 124):
            halt(binding, 'author process failed before normal cutoff', 'authoring')
            failure = dict(phase='authoring', type='AuthorExit', exit_code=model_exit,
                           reason='author process failed before the normal search cutoff')
        best = nominee(root, binding['compiler'])
        if best:
            phase = 'confirmation'
            validate_nominee(root, binding, best, environment, deadline['stop_at_epoch'])
            seconds = int(deadline['stop_at_epoch'] - time.time()) - 20
            if seconds <= 0:
                raise TimeoutError('confirmation budget exhausted')
            command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(seconds) + 's',
                'python3', 'campaign/development_evaluate_owner.py', '--candidate', best['candidate'],
                '--id', 'final-confirmation', '--phase', 'confirmation']
            with (root / 'campaign/logs/confirmation-owner.log').open('x') as log:
                code = subprocess.call(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=environment)
            path = root / 'campaign/evaluations/final-confirmation/outcome.json'
            confirmation = json.loads(path.read_text()) if path.exists() else dict(status='missing', exit_code=code)
    except Exception as error:
        if isinstance(error, (OSError, subprocess.SubprocessError)):
            halt(binding, 'controller service or observer failed: ' + type(error).__name__, phase)
        failure = dict(phase=phase, type=type(error).__name__, reason=str(error))
    write_terminal(root, binding, model_exit, best, confirmation, failure, closure)
    return 1 if failure else 0


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['--check']):
        raise SystemExit('development_launch takes only optional --check')
    raise SystemExit(main(check=sys.argv[1:] == ['--check']))
